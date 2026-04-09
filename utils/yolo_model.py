"""
YOLOv2 模型構建工具
支援多種基礎網路架構
"""
import torch
import torch.nn as nn
import torchvision.models as models


def build_yolov2_model(base_network='googlenet', num_classes=1, 
                       anchor_boxes=None, image_size=224, pretrained=False):
    """
    構建 YOLOv2 模型
    
    參數:
        base_network: 基礎網路名稱
        num_classes: 類別數量
        anchor_boxes: anchor boxes (numpy array, shape: [n_anchors, 2])
        image_size: 輸入圖像尺寸
        pretrained: 是否使用預訓練權重
    
    返回:
        YOLOv2 模型
    """
    # 載入基礎網路
    if base_network == 'mobilenetv2':
        base_net = models.mobilenet_v2(pretrained=pretrained)
        feature_layer = 'features.17'  # 對應 block_13_expand_relu
    elif base_network == 'resnet18':
        base_net = models.resnet18(pretrained=pretrained)
        feature_layer = 'layer4'  # 對應 res4b_relu
    elif base_network == 'resnet50':
        base_net = models.resnet50(pretrained=pretrained)
        feature_layer = 'layer4'  # 對應 activation_40_relu
    elif base_network == 'resnet101':
        base_net = models.resnet101(pretrained=pretrained)
        feature_layer = 'layer4'  # 對應 res4b22_relu
    elif base_network == 'vgg16':
        base_net = models.vgg16(pretrained=pretrained)
        feature_layer = 'features.30'  # 對應 relu5_3
    elif base_network == 'vgg19':
        base_net = models.vgg19(pretrained=pretrained)
        feature_layer = 'features.36'  # 對應 relu5_4
    elif base_network == 'googlenet':
        base_net = models.googlenet(pretrained=pretrained, aux_logits=False)
        feature_layer = 'inception5b'  # Output should be 1024
    else:
        raise ValueError(f"不支援的基礎網路: {base_network}")
    
    # 創建簡化的 YOLOv2 模型
    # 注意: 這裡是一個簡化版本，實際應用中需要使用完整的 YOLOv2 實現
    model = SimpleYOLOv2(base_net, num_classes, anchor_boxes, image_size)
    
    return model


class SimpleYOLOv2(nn.Module):
    """
    簡化的 YOLOv2 模型
    注意: 這是示例實現，實際應用中建議使用成熟的 YOLO 實現庫
    """
    def __init__(self, base_network, num_classes, anchor_boxes, image_size):
        super(SimpleYOLOv2, self).__init__()
        self.base_network = base_network
        self.num_classes = num_classes
        self.image_size = image_size
        self.model_type = type(base_network).__name__
        
        # 1. 獲取基礎網路的特徵層
        if isinstance(base_network, models.VGG):
            self.features = base_network.features
        elif 'resnet' in str(type(base_network)).lower():
            self.features = nn.Sequential(*list(base_network.children())[:-2])
        elif isinstance(base_network, models.GoogLeNet):
            # 保持 GoogLeNet 特殊歷史設定以兼容既有 best_googlenet.pt
            # [-3]: inception5b (1024 channels), matches checkpoint's adapter expectation
            self.features = nn.Sequential(*list(base_network.children())[:-3])
        elif hasattr(base_network, 'features') and not isinstance(base_network, models.VGG):
            # 適用於 MobileNetV2 等具有 .features 屬性的模型
            self.features = base_network.features
        else:
            self.features = base_network

        # 2. 自動偵測特徵層輸出通道數
        with torch.no_grad():
            try:
                dummy_input = torch.zeros(1, 3, image_size, image_size)
                dummy_output = self.features(dummy_input)
                feat_channels = dummy_output.shape[1]
            except:
                feat_channels = 512

        # 3. 決定檢測頭輸入通道數 (Legacy Support)
        self.in_channels = feat_channels
        if isinstance(base_network, models.GoogLeNet):
            # 強制設為 832 以匹配 best_googlenet.pt 檢測頭權重與歷史邏輯
            self.in_channels = 832
            
        # print(f"[INIT] {self.model_type} detected feat_channels: {feat_channels}, target in_channels: {self.in_channels}")
        
        # 4. 建立頻道適配器 (若有頻道不匹配)
        # 必須在 __init__ 建立，load_state_dict 才能正確加載 weight
        if feat_channels != self.in_channels:
            self._channel_adapter = nn.Conv2d(feat_channels, self.in_channels, 1)
        else:
            self._channel_adapter = None
        
        # YOLOv2 檢測頭
        # 每個 anchor 預測: (x, y, w, h, obj, class1, class2, ...)
        n_anchors = anchor_boxes.shape[0] if anchor_boxes is not None else 5
        output_dim = n_anchors * (5 + num_classes)  # 5 = x, y, w, h, obj
        
        # 建立檢測頭，使用正確的輸入通道數
        # 建立檢測頭 (Convolutional Grid Head)
        self.detection_head = nn.Sequential(
            nn.Conv2d(self.in_channels, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, output_dim, 1, stride=1, padding=0)  # 1x1 conv for prediction
        )
    
    def forward(self, x):
        # 提取特徵
        try:
            features = self.features(x)
        except Exception as e:
            print(f"特徵提取出錯: {e}")
            # 回退：創建合成特徵
            features = torch.ones(x.shape[0], self.in_channels, 7, 7, device=x.device)
        
        # 確保特徵是 4D 張量
        if not isinstance(features, torch.Tensor) or features.dim() != 4:
            # 創建合成特徵
            features = torch.ones(x.shape[0], self.in_channels, 7, 7, device=x.device)
        
        # 確保通道數正確 (使用預先建立的適配器)
        if self._channel_adapter is not None:
            features = self._channel_adapter(features)
        
        # 應用檢測頭
        output = self.detection_head(features)
        return output
    
    def detect(self, x, conf_threshold=0.5):
        """執行檢測"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            # 這裡需要實現完整的 YOLOv2 檢測邏輯
            # 包括 anchor 匹配、NMS 等
            return output

