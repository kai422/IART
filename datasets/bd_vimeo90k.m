
% Define the main folder and output folder
mainFolder = '/home/kai422/vimeo_septuplet/sequences';
outputFolder = '/home/kai422/vimeo_septuplet/sequences_bd'; % Replace with your desired output path

% Parameters for blur-downsampled degradation
scale = 4;
sigma = 1.6;
kernelsize = ceil(sigma * 3) * 2 + 2;
kernel = fspecial('gaussian', kernelsize, sigma);

% Get a list of all subfolders
subfolders = dir(mainFolder);
subfolders = subfolders([subfolders.isdir]); % Filter only directories
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Remove '.' and '..'

% Process each image in each subfolder
for i = 1:length(subfolders)
    subfolderName = subfolders(i).name;
    subSubfolders = dir(fullfile(mainFolder, subfolderName));
    subSubfolders = subSubfolders([subSubfolders.isdir]); % Filter only directories
    subSubfolders = subSubfolders(~ismember({subSubfolders.name}, {'.', '..'})); % Remove '.' and '..'
    
    for k = 1:length(subSubfolders)
        subSubfolderName = subSubfolders(k).name;
        imageFiles = dir(fullfile(mainFolder, subfolderName, subSubfolderName, '*.png')); % Assuming images are PNG, change if different

        % Create corresponding subfolder in the output folder
        outputSubfolder = fullfile(outputFolder, subfolderName, subSubfolderName);
        if ~exist(outputSubfolder, 'dir')
            mkdir(outputSubfolder);
        end

        for j = 1:length(imageFiles)
            % Read the image
            fileName = imageFiles(j).name;
            disp(fullfile(mainFolder, subfolderName, subSubfolderName, fileName))
            gt = imread(fullfile(mainFolder, subfolderName, subSubfolderName, fileName));

            % Check if the image is grayscale and convert it to 3 channels if necessary
            if size(gt, 3) == 1
                gt = cat(3, gt, gt, gt);
            end

            % Apply blur and downsample
            lq = imfilter(gt, kernel, 'replicate');
            lq = lq(scale/2:scale:end-scale/2, scale/2:scale:end-scale/2, :);

            % Save the processed image in the corresponding subfolder
            [~, name, ext] = fileparts(fileName);
            outputFileName = fullfile(outputSubfolder, [name, ext]);
            imwrite(lq, outputFileName);
        end
    end
end
