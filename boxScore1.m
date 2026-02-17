box_n=0; candi_n=0;  cnt_bat=0; desiredBoxN=0;
for i=1:length(trainingData1.imageFilename)
	labelBox=trainingData1.vehicle{i};
	desiredBoxN=desiredBoxN+size(labelBox,1);
	img=imread(trainingData1.imageFilename{i});
	try
		[bbox, ~, ~] = detect(detector, img);
		%[bbox,scores] = detect(detector,img);
		%cb=(cnt_bat+1):(cnt_bat+size(scores,1));
		%c_b=cb';
		%cnt_bat=cnt_bat+size(scores,1);
		%I = insertObjectAnnotation(img,'rectangle',bbox,c_b);
		%I = insertObjectAnnotation(img,'rectangle',bbox,scores);
		%I = insertObjectAnnotation(img,'rectangle',bbox,0);
		%figure(i),imshow(I);
		%title('Detected capacitor and Detection Scores');
		tbox=[];
		candi_n=candi_n+size(bbox,1);
		for ib=1:size(bbox,1)
			[err,~]=min(sum(abs([repmat(bbox(ib,1),size(labelBox,1),1)-labelBox(:,1),...
			repmat(bbox(ib,2),size(labelBox,1),1)-labelBox(:,2)]),2));
			err_base=min([bbox(ib,3),bbox(ib,4)]);
			if err(1)<(err_base*0.95)  % 0.7: 417 ;  0.75: 429
				 tbox=[tbox;bbox(ib,:)];
			end
		end
		cb=(cnt_bat+1):(cnt_bat+size(tbox,1));
		c_b=cb';
		cnt_bat=cnt_bat+size(tbox,1);
		%I = insertObjectAnnotation(img,'rectangle',tbox,c_b);
		%I = insertObjectAnnotation(img,'rectangle',tbox,0);
		%figure(i),imshow(I);
		box_n=box_n+size(tbox,1);
	catch
	    % isempty(bbox)
	end
end