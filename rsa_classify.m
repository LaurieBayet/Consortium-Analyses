function [classification, comparisons] = rsa_classify(model_data, model_labels, test_data, test_labels, opts)
%% rsa_classify implements a correlation-based, similarity-space classifier
% following Zinszer, Bayet, Emberson, Raizada, & Aslin's (2018, Neurophotonics)
% method for all-possible-pairwise-comparisons and Zinszer, Anderson, Kang,
% Wheatley, & Raizada's (2016) approach for n-way comparison
%
% Correlation statistic (pearson, spearman, or kendall) may be selected
% using opts.corr_stat, e.g., default is opts.corr_stat='spearman'
%
% Only 'exclusive' mode is available.
%
% If opts.tiebreak is set to true (default), the correlation coefficients
% are randomly adjusted by <1% of the smallest observed difference to
% prevent exact matches and thus prevent ties in the classification. If
% opts.tiebreak is set to false, the classifier will prefer classes
% appearing earlier in the training set.
%
% If opts.verbose is set to true (default is false), Fisher-adjusted
% (hyperbolic arctangent) correlation matrices are displayed for all
% subjects, sessions, etc.
%
% All-possible-pairwise comparison (opts.pairwise) is under development.

%% If the options struct is not provided, set default parameters
if ~exist('opts','var') || isempty(opts)
    opts = struct;
    opts.corr_stat = 'spearman';
    opts.exclusive = true;
    opts.pairwise = false;
    opts.tiebreak = true;
    opts.verbose = 0;
end

% Pull a list of all the unique classes / conditions, preserving order
model_classes = unique(model_labels(:),'stable');

%% check to see the orders of test and train data - this is to see if they match
% get number of categories

[~,train_order] = sort(model_labels);
[~,test_order] = sort(test_labels);

%% sort test data based on the re-ordered data
test_labs = test_labels(test_order);
test_dat = test_data(test_order, :,:,:);

model_labs = model_labels(train_order);
model_dat = model_data(train_order, :,:,:);


%% Build similarity structures
% Transform the model_data into similiarty structures for each session by
% correlating between conditions and then average the similarity structures
% together into a single model.

%  iterate through all the layers (3rd dimension) and create
% correlation matrices

model_correl = nan(size(model_dat,1),size(model_dat,1),size(model_data,3),size(model_dat,4));

for i = 1: (size(model_dat,3)*size(model_dat,4))
    model_correl(:,:,i) = corr(model_dat(:,:,i)', 'type', opts.corr_stat);
end
training_matrix = nanmean(model_correl,3);
training_matrix = nanmean(training_matrix,4);

%  iterate through all the layers (3rd dimension) and create
% correlation matrices

test_correl = nan(size(test_dat,1),size(test_dat,1),size(test_dat,3),size(test_dat,4));
for i = 1: (size(test_dat,3)*size(test_dat,4))
    test_correl(:,:,i) = corr(test_dat(:,:,i)', 'type', opts.corr_stat);
end
test_matrix = nanmean(test_correl,3);
test_matrix = nanmean(test_matrix,4);

%% Visualize the matrices

if opts.verbose > 1
    
    plot_idx = 1;
    figure
    for session_idx = 1:size(model_data,3)
        for subject_idx = 1:size(model_data,4)
            subplot(size(model_data,3),size(model_data,4),plot_idx);
            imagesc(model_correl(:,:,session_idx, subject_idx))
            title(['Subj ' num2str(subject_idx) ' Sess ' num2str(session_idx)])
            xticklabels([])
            yticklabels([])
            caxis([-.5,.5])
            colorbar('hot')
            plot_idx = plot_idx + 1;
        end
    end
    
    plot_idx = 1;
    figure
    for session_idx = 1:size(test_data,3)
        for subject_idx = 1:size(test_data,4)
            subplot(size(test_data,3),size(test_data,4),plot_idx);
            imagesc(test_correl(:,:,session_idx, subject_idx))
            title(['Subject ' num2str(subject_idx) ' Session ' num2str(session_idx)])
            xticklabels([])
            yticklabels([])
            caxis([-.5,.5])
            colorbar('hot')
            plot_idx = plot_idx + 1;
        end
    end
    
end
if opts.verbose
    figure();
    % Training data
    subplot(1,2,1)
    imagesc(training_matrix)
    title('Training Data')
    xticklabels(model_labels)
    yticklabels(model_labels)
    %caxis([min(min(tril(training_matrix,-1))),max(max(tril(training_matrix,-1)))])
    caxis([-.5,.5])
    
    colorbar('hot')
    [i, j, ~] = find(~isnan(training_matrix));
    text(i-.3,j,num2str(round(training_matrix(~isnan(training_matrix)),2)));
    
    % Test data
    subplot(1,2,2)
    imagesc(test_matrix)
    title('Test Data')
    xticklabels(test_labels)
    yticklabels(test_labels)
    %caxis([min(min(tril(test_matrix,-1))),max(max(tril(test_matrix,-1)))])
    caxis([-.5,.5])
    colorbar('hot')
    [i, j, ~] = find(~isnan(test_matrix));
    text(i-.3,j,num2str(round(test_matrix(~isnan(test_matrix)),2)));
end

%% Sanity Check
if sum(isnan(test_matrix(:)))==numel(test_matrix) || sum(isnan(training_matrix(:)))==numel(training_matrix)
    error('One or both input matrices contains all NaN values. I quit!');
end

%% Save out the classification results based on greatest correlation coefficient for each test pattern
% Initialize empty cell matrix for classifications

if ~isfield(opts,'pairwise') || ~opts.pairwise
    
    % Number of stimulus classes to compare, based on the number of rows
    number_classes = size(test_matrix,1);
    
    % Too many permutations if more than 10 classes. Throw error and leave.
    if number_classes>10, error('n-way classification does not work for more than 10 classes. Use opts.pairwise=true'); end
    
    % Generate a list of every pairwise comparison and the results array
    list_of_comparisons = perms(1:number_classes);
    number_of_comparisons = size(list_of_comparisons,1);
    results_of_comparisons = nan(number_of_comparisons,1);
    
    % Convert training set to a vector for the comparisons
    train_vec = training_matrix(logical(tril(ones(size(training_matrix)),-1)));
    
    % Iterate through the different Permutations of test
    for perm_idx = 1:number_of_comparisons
        tmp_test_matrix = test_matrix(list_of_comparisons(perm_idx,:),list_of_comparisons(perm_idx,:));
        tmp_test_vec = tmp_test_matrix(logical(tril(ones(size(tmp_test_matrix)),-1)));
        results_of_comparisons(perm_idx) = corr(atanh(train_vec),atanh(tmp_test_vec),'rows','pairwise');
    end
    
    % Choose the best of all the permutations of labels
    [rating, best_perm] = max(results_of_comparisons);
    classification = model_classes(list_of_comparisons(best_perm,:));
    
    %accuracy = strcmp(classification,test_labels);
    comparisons = test_labels';
    
    % put the labels back in the order they were put in as
    [~,reorder_test] = sort(test_order);
    classification = classification(reorder_test);
    comparisons = comparisons(reorder_test);
    
    
else
    [accuracy, comparisons] = pairwise_rsa_test(atanh(test_matrix),atanh(training_matrix));
    classification = comparisons;
    classification(~accuracy,:) = classification(~accuracy,end:-1:1);
    classification = model_classes(classification);
    comparisons = model_classes(comparisons);
    
    results_of_comparisons = cell(size(classification,2), 2, size(classification,1));
    for comp = 1:size(classification,1)
        results_of_comparisons(:,1,comp) = classification(comp,:)';
        results_of_comparisons(:,2,comp) = comparisons(comp,:)';
    end
    
    classification = results_of_comparisons;
    
end

% put the labels back in the order they were put in as
[~,reorder_test] = sort(test_order);
classification = classification(reorder_test);

end
