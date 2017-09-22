%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lauren Howard - W1287305
% COEN 240 - Coding Assignment 5
% pca.m

% This script uses bayes decision theory on a reduced dimension dataset of the
% iris dataset. The first run naively reduces the dimensions by using only the
% first dimension of the iris dataset. The second run uses the first principal
% component using PCA. The second run has more accurate results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;
data_raw = dlmread('iris_dataset.dat');

N = 150;  % total number of samples
D = 1; % num features
NC = 50;  % size of each class
K = 10;  % K-fold
NK = N/K; % Number of examples per fold

% Randomly shuffle data
index = randperm(N);
shuffled_data = data_raw(index,:);

X = shuffled_data(:,1:4);
y = shuffled_data(:,5);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USING THE FIRST DIMENSION OF X, NO PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = [X(:,1) y];

% the code below is from the third programming assignment
total_correct_sum = 0;
iterations_correct_vec = [];

for k=1:K
  % the start/end index of the fold
  test_start_index = (k-1)*NK + 1;
  test_end_index = (test_start_index+NK) - 1;

  % test data comes from the fold
  test_data = data(test_start_index:test_end_index,:);

  % train data is the non-fold data
  train_data = vertcat(
    data(1:test_start_index-1,:),
    data(test_end_index+1:N,:)
  );

  % filter the training data based on class
  class_1_data = train_data(train_data(:,D+1) == 1,:);
  class_2_data = train_data(train_data(:,D+1) == 2,:);
  class_3_data = train_data(train_data(:,D+1) == 3,:);

  % get the features for each class-data-set
  class_1_features = class_1_data(:,1:D);
  class_2_features = class_2_data(:,1:D);
  class_3_features = class_3_data(:,1:D);

  % compute the mean of each feature for that class
  u1 = (sum(class_1_features)/length(class_1_features)).';
  u2 = (sum(class_2_features)/length(class_2_features)).';
  u3 = (sum(class_3_features)/length(class_3_features)).';

  % compute covariance mat for the features
  cov1 = covmle(class_1_features, u1);
  cov2 = covmle(class_2_features, u2);
  cov3 = covmle(class_3_features, u3);

  % the sum of 'correct classifications'
  current_correct_sum = 0;

  % iterate through the test data to compute accuracy
  for i=1:NK
    % the test vector
    x = test_data(i,1:D);
    % the actual class of this example
    class = test_data(i,D+1);

    % computes the probability from the discriminants
    xm1 = x.' - u1;
    g1 = -0.5 * xm1.' * inv(cov1) * xm1;

    xm2 = x.' - u2;
    g2 = -0.5 * xm2.' * inv(cov2) * xm2;

    xm3 = x.' - u3;
    g3 = -0.5 * xm3.' * inv(cov3) * xm3;

    % gets the class label of the highest discriminant
    [_,predicted_class] = max([g1, g2, g3]);

    % if the class label is correct then increment the counters
    if predicted_class == class
      current_correct_sum += 1;
      total_correct_sum += 1;
    end
  end

  % iterations_correct_vec contains the sum of correct classifications for
  % each iteration of the 10-fold
  iterations_correct_vec = vertcat(
    iterations_correct_vec, 
    current_correct_sum
  );

end

fprintf('Accuracy per iteration =\n');
disp(iterations_correct_vec/NK);
fprintf('Total Accuracy = %5.4f\n', total_correct_sum/N);







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% USING THE FIRST PRINCIPAL COMPONENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = repmat(mean(X), length(X), 1);
XM = X-M;

% dimensionality reduction
[eigvec_org eigval_org] = eig(cov(XM));
eigvec = fliplr(eigvec_org);
eigval = flipud(diag(eigval_org));

fprintf('\neigval =\n');
disp(eigval);
fprintf('\n');

% principal companents mat
PC = XM*eigvec;

% we are only supposed to use the first pricinpal component
data = [PC(:,1) y];

% the code below is from the third programming assignment
total_correct_sum = 0;
iterations_correct_vec = [];

for k=1:K
  % the start/end index of the fold
  test_start_index = (k-1)*NK + 1;
  test_end_index = (test_start_index+NK) - 1;

  % test data comes from the fold
  test_data = data(test_start_index:test_end_index,:);

  % train data is the non-fold data
  train_data = vertcat(
    data(1:test_start_index-1,:),
    data(test_end_index+1:N,:)
  );

  % filter the training data based on class
  class_1_data = train_data(train_data(:,D+1) == 1,:);
  class_2_data = train_data(train_data(:,D+1) == 2,:);
  class_3_data = train_data(train_data(:,D+1) == 3,:);

  % get the features for each class-data-set
  class_1_features = class_1_data(:,1:D);
  class_2_features = class_2_data(:,1:D);
  class_3_features = class_3_data(:,1:D);

  % compute the mean of each feature for that class
  u1 = (sum(class_1_features)/length(class_1_features)).';
  u2 = (sum(class_2_features)/length(class_2_features)).';
  u3 = (sum(class_3_features)/length(class_3_features)).';

  % compute covariance mat for the features
  cov1 = covmle(class_1_features, u1);
  cov2 = covmle(class_2_features, u2);
  cov3 = covmle(class_3_features, u3);

  % the sum of 'correct classifications'
  current_correct_sum = 0;

  % iterate through the test data to compute accuracy
  for i=1:NK
    % the test vector
    x = test_data(i,1:D);
    % the actual class of this example
    class = test_data(i,D+1);

    % computes the probability from the discriminants
    xm1 = x.' - u1;
    g1 = -0.5 * xm1.' * inv(cov1) * xm1;

    xm2 = x.' - u2;
    g2 = -0.5 * xm2.' * inv(cov2) * xm2;

    xm3 = x.' - u3;
    g3 = -0.5 * xm3.' * inv(cov3) * xm3;

    % gets the class label of the highest discriminant
    [_,predicted_class] = max([g1, g2, g3]);

    % if the class label is correct then increment the counters
    if predicted_class == class
      current_correct_sum += 1;
      total_correct_sum += 1;
    end
  end

  % iterations_correct_vec contains the sum of correct classifications for
  % each iteration of the 10-fold
  iterations_correct_vec = vertcat(
    iterations_correct_vec, 
    current_correct_sum
  );

end

fprintf('Accuracy per iteration =\n');
disp(iterations_correct_vec/NK);
fprintf('Total Accuracy = %5.4f\n', total_correct_sum/N);
