clc, clearvars, close all, format compact

%% Read data
SDNDataSet = readtable("DDoS Attack SDN Dataset.csv");

%% Normal and Attack count
labelVector = SDNDataSet.label;
[labelCount, labelValues] = groupcounts(labelVector);
labels = {'Normal', 'Attack'};
pie(labelCount);
legendPie = legend(labels);

%% Find missing value
nanCount = sum(ismissing(SDNDataSet));
% Sort the columns based on the count of NaN values
[sortedCounts, sortedIndices] = sort(nanCount);
% Create a bar plot to visualize the counts of NaN values
bar(sortedCounts);
% Label the x-axis with column names
xticks(1:numel(sortedCounts));
xticklabels(SDNDataSet.Properties.VariableNames(sortedIndices));
% Label the y-axis
ylabel('Count of NaN Values');
% Add a title
title('NaN Value Counts for Columns in the DDoS Attack SDN Dataset');

% Find numeric and object columns
% Initialize arrays to store column names for numeric and object columns
numericColumns = {};
countNumCol = 0;
objectColumns = {};
countObjCol = 0;

% Loop through each column in the table
for i = 1:size(SDNDataSet, 2)
    column = SDNDataSet{:, i};
    if isnumeric(column)
        numericColumns = [numericColumns, SDNDataSet.Properties.VariableNames{i}];
        countNumCol = countNumCol + 1;
    elseif iscell(column) && ischar(column{1})
        objectColumns = [objectColumns, SDNDataSet.Properties.VariableNames{i}];
        countObjCol = countObjCol + 1;
    end
end

fprintf('Number of Numeric values: %d\n', countNumCol);
fprintf('Number of Object values: %d\n', countObjCol);


%% Graph
% Exclude specific non-numeric columns ('dt' and 'src')
excludedColumns = {'dt', 'src'};
numericColumnNames = SDNDataSet.Properties.VariableNames(~ismember(SDNDataSet.Properties.VariableNames, excludedColumns));
% Create figure 1
figure('Position', [0, 0, 800, 500]);

% Convert 'src' column to a cell array
srcCell = table2cell(SDNDataSet(:, 'src'));

% Get unique IP addresses and their counts for all requests
[allIPAddresses, ~, idx] = unique(srcCell);
allIPCounts = histcounts(idx, 1:numel(allIPAddresses) + 1);

% Create horizontal bar chart for all IP addresses (color: lawngreen)
barh(allIPCounts, 'FaceColor', [0.5, 0.9, 0.5]);

% Add labels for all IP addresses
for idx = 1:length(allIPAddresses)
    text(allIPCounts(idx), idx - 0.2, num2str(allIPCounts(idx)), 'Color', 'r', 'FontSize', 13);
end

hold on;

% Filter data for malicious requests (label == 1)
maliciousData = SDNDataSet(SDNDataSet.label == 1, :);

% Convert 'src' column to a cell array for malicious data
maliciousSrcCell = table2cell(maliciousData(:, 'src'));

% Get unique IP addresses and their counts for malicious requests
[maliciousIPAddresses, ~, maliciousIdx] = unique(maliciousSrcCell);
maliciousIPCounts = histcounts(maliciousIdx, 1:numel(maliciousIPAddresses) + 1);

% Create horizontal bar chart for malicious IP addresses (color: blue)
barh(maliciousIPCounts, 'FaceColor', 'b');

% Add labels for malicious IP addresses
for idx = 1:length(maliciousIPAddresses)
    text(maliciousIPCounts(idx), idx - 0.2, num2str(maliciousIPCounts(idx)), 'Color', 'w', 'FontSize', 13);
end

hold off;

% Set axis labels, legend, and title
xlabel('Number of Requests');
ylabel('IP Address of Sender');
legend('All', 'Malicious');
title('Number of Requests from Different IP Addresses');

% Figure 2
% Create a figure
figure('Position', [0, 0, 800, 500]);

% Get the counts of protocols for all requests
allProtocolCounts = countcats(categorical(SDNDataSet.Protocol));
% Get the counts of protocols for malicious requests
maliciousProtocolCounts = countcats(categorical(SDNDataSet(SDNDataSet.label == 1, :).Protocol));

% Define the x-axis values (protocol labels)
protocolLabels = categories(categorical(SDNDataSet.Protocol));

% Create a bar chart for all protocols (color: red)
bar(allProtocolCounts, 'FaceColor', 'r');
hold on;

% Create a bar chart for malicious protocols (color: blue)
bar(maliciousProtocolCounts, 'FaceColor', 'b');

% Add labels for all protocols
for idx = 1:length(protocolLabels)
    text(idx - 0.15, allProtocolCounts(idx) + 200, num2str(allProtocolCounts(idx)), 'Color', 'black', 'FontSize', 17);
end

% Add labels for malicious protocols
for idx = 1:length(protocolLabels)
    text(idx - 0.15, maliciousProtocolCounts(idx) + 200, num2str(maliciousProtocolCounts(idx)), 'Color', 'w', 'FontSize', 17);
end

hold off;

% Set x-axis labels (protocols)
xticks(1:numel(protocolLabels));
xticklabels(protocolLabels);

% Set axis labels, legend, and title
xlabel('Protocol');
ylabel('Count');
legend('All', 'Malicious');
title('The Number of Requests from Different Protocols');

%% Remove rows with missing values
SDNDataSet = SDNDataSet(~any(ismissing(SDNDataSet), 2), :);

%% Standardize numeric columns
for i = 1:numel(numericColumns)
    col = numericColumns{i};
    SDNDataSet.(col) = zscore(SDNDataSet.(col));
end

%% Split the dataset into training and testing sets
rng(123); % Set a random seed for reproducibility
splitRatio = 0.8; % 80% for training, 20% for testing
numRows = size(SDNDataSet, 1);
idx = randperm(numRows);
trainIdx = idx(1:round(splitRatio * numRows));
testIdx = idx(round(splitRatio * numRows) + 1:end);
trainData = SDNDataSet(trainIdx, :);
testData = SDNDataSet(testIdx, :);

% save training and testing set
save('preprocessed_training_set.mat', 'trainData');
save('preprocessed_testing_set.mat', 'testData');

%% Autoencoder
% Load the preprocessed data
trainData = load('preprocessed_training_set.mat');

% Exclude non-numeric columns (e.g., 'dt' and 'src')
X_train = trainData.trainData{:,numericColumns};

% Define the architecture of the autoencoder
inputSize = size(X_train, 2);
hiddenSize = 100;
outputSize = inputSize;

% Create and configure the autoencoder
autoenc = trainAutoencoder(X_train', hiddenSize, ...
    'MaxEpochs', 100, ...
    'L2WeightRegularization', 0.001, ... % Regularization parameter
    'SparsityRegularization', 4, ... % Sparsity regularization parameter
    'SparsityProportion', 0.05, ... % Sparsity proportion
    'ScaleData', false);

% Save autoencoder
save('trained_autoencoder.mat', 'autoenc');

% Encode the data using the trained autoencoder
encodedData = encode(autoenc, X_train');

%% Test
% Load the testing data
testData = load('preprocessed_testing_set.mat');
X_test = testData.testData{:, numericColumns};

% Encode the testing data using the same autoencoder
encodedTestData = encode(autoenc, X_test');

% Train a classifier using the encoded data
svmClassifier = fitcsvm(encodedData', trainData.trainData.label, 'KernelFunction', 'linear');

% Make predictions using the SVM classifier
predictions = predict(svmClassifier, encodedTestData');

% Evaluate the classifier's performance (e.g., accuracy, confusion matrix, etc.)
accuracy = sum(predictions == testData.testData.label) / length(predictions);
confusionMatrix = confusionmat(testData.testData.label, predictions);

% Decode the encoded data to obtain the reconstructed data
reconstructedData = decode(autoenc, encodedData);

% Calculate the reconstruction error
reconstructionError = sum((X_train - reconstructedData').^2, 2); % Mean squared error

% Visualize the reconstruction error (histogram)
figure;
hist(reconstructionError, 50);
xlabel('Reconstruction Error');
ylabel('Frequency');
title('Reconstruction Error Distribution');

% threshold
threshold = 30;
anomalies = sum(reconstructionError > threshold);
totalDataPoints = length(reconstructionError);

% Calculate the percentage of anomalies
percentageAnomalies = anomalies / totalDataPoints * 100;

disp(['Percentage of Anomalies: ', num2str(percentageAnomalies), '%']);

% Display the results
disp(['Accuracy: ', num2str(accuracy)]);
disp('Confusion Matrix:');
disp(confusionMatrix);

%% Other performance metrics
% Compute the confusion matrix
confusionMatrix = confusionmat(testData.testData.label, predictions);

% Extract true positives, false positives, and false negatives
truePositives = confusionMatrix(2, 2);
falsePositives = confusionMatrix(1, 2);
falseNegatives = confusionMatrix(2, 1);

% Calculate precision, recall, and F1 score
precision = truePositives / (truePositives + falsePositives);
recall = truePositives / (truePositives + falseNegatives);
f1Score = 2 * (precision * recall) / (precision + recall);

fprintf('Precision: %.4f\n', precision);
fprintf('F1 Score: %.4f\n', f1Score);


