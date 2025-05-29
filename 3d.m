clc;
clear;
close all;

% Load LiDAR data
filename = 'sample.ply';
ptCloudOrig = pcread(filename);

% Visualize Original
figure('Name','Original Point Cloud');
pcshow(ptCloudOrig);
title('Original LiDAR Point Cloud');
xlabel('X'); ylabel('Y'); zlabel('Z');

% Downsample
ptCloudDS = pcdownsample(ptCloudOrig, 'gridAverage', 0.05);

% Denoise
ptCloudDenoised = pcdenoise(ptCloudDS, 'Threshold', 1);

% Ground Segmentation
maxDistance = 0.2;
[model, inlierIdx, outlierIdx] = pcfitplane(ptCloudDenoised, maxDistance);
ground = select(ptCloudDenoised, inlierIdx);
objects = select(ptCloudDenoised, outlierIdx);

% Clustering
minDist = 0.5; % object separation threshold
[labels, numClusters] = pcsegdist(objects, minDist);
cmap = lines(numClusters);

% Init output table
objectData = [];

% Visualize Clusters + Bounding Boxes
figure('Name','Detected Objects with Bounding Boxes');
pcshow(ground.Location, 'g');
hold on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Object Detection from LiDAR');
axis equal;

for i = 1:numClusters
    % Select cluster
    idx = (labels == i);
    clusterPC = select(objects, idx);
    
    % Get cluster properties
    clusterPoints = clusterPC.Location;
    centroid = mean(clusterPoints, 1);
    numPoints = clusterPC.Count;

    % Fit bounding box
    xMin = min(clusterPoints(:,1));
    xMax = max(clusterPoints(:,1));
    yMin = min(clusterPoints(:,2));
    yMax = max(clusterPoints(:,2));
    zMin = min(clusterPoints(:,3));
    zMax = max(clusterPoints(:,3));
    
    % Store in output
    objectData = [objectData; i, centroid, numPoints];

    % Visualize colored cluster
    pcshow(clusterPC.Location, cmap(i,:), 'MarkerSize', 30);
    
    % Draw bounding box
    plot3([xMin xMax xMax xMin xMin], [yMin yMin yMax yMax yMin], [zMin zMin zMin zMin zMin], 'k-','LineWidth',1.5);
    plot3([xMin xMax xMax xMin xMin], [yMin yMin yMax yMax yMin], [zMax zMax zMax zMax zMax], 'k-','LineWidth',1.5);
    for j = 0:3
        plot3([xMin xMin], [yMin yMin], [zMin zMax], 'k--','LineWidth',1.0);
        xMin = xMin + (xMax - xMin)/3;
        yMin = yMin + (yMax - yMin)/3;
    end

    % Show ID at centroid
    text(centroid(1), centroid(2), centroid(3)+0.2, ['Obj ' num2str(i)], 'Color','k', 'FontSize',8);
end

legend({'Ground', 'Detected Objects'});

% Export to CSV
T = array2table(objectData, 'VariableNames', ...
    {'ID', 'CentroidX', 'CentroidY', 'CentroidZ', 'NumPoints'});
writetable(T, 'detected_objects.csv');

disp('✅ Object detection complete.');
disp(T);
