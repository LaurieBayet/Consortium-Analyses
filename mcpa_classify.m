function [classification, rating] = mcpa_classify(model_data, model_labels, test_data, test_labels, opts)
%% mcpa_classify implements a correlation-based, channel-space classifier
% following Emberson, Zinszer, Raizada & Aslin's (2017, PLoS One) method
% and extending this approach to multiple conditions (>2). The MCPA
% approach treats channels as features and compares the correlation
% coefficients between classes in the training data (averaged across all
% examples) and instances in the test data.
%
% Correlation statistic (pearson, spearman, or kendall) may be selected
% using opts.corr_stat, e.g., opts.corr_stat='pearson'
%
% In 'non-exclusive' mode (the default), classification decisions are based
% simply on the greatest Fisher-adjusted correlation between test instance
% and classes in training set.
% In 'exclusive' mode (opts.exclusive=true), the pairwise matches between
% test classes are training classes are optimized for greatest sum of
% Fisher-adjusted correlations.
%
% If opts.tiebreak is set to true (default), the correlation coefficients
% are adjusted by <1% of the smallest observed difference to prevent exact
% matches and thus prevent ties in the classification. If opts.tiebreak is
% set to false, the classifier will prefer classes appearing earlier in the
% training set.
%
% All-possible-pairwise comparison (opts.pairwise) is under development.

%% If the options struct is not provided, set default parameters
if ~exist('opts','var') || isempty(opts) 
    opts = struct;
end

if ~isfield(opts, 'corr_stat')
    opts.corr_stat = 'spearman';
end
if ~isfield(opts, 'exclusive')
    opts.exclusive = false;
end
if ~isfield(opts, 'pairwise')
    opts.pairwise = false;
end
if ~isfield(opts, 'tiebreak')
    opts.tiebreak = true;
end


model_classes = unique(model_labels,'stable');

%% check to see the orders of test and train data - this is to see if they match
% get number of categories

[~,train_order] = sort(model_labels);
[~,test_order] = sort(test_labels);

%% sort test data based on the re-ordered data
test_labs = test_labels(test_order);
test_dat = test_data(test_order, :);

model_labs = model_labels(train_order);
model_dat = model_data(train_order, :);

%% Average across training data to get model features for each class
model_patterns = nan(size(model_data,2),length(model_classes));
for class_idx = 1:length(model_classes)
    model_patterns(:,class_idx) = nanmean(model_dat(strcmp(model_classes{class_idx},model_labs),:),1)';
end

%% Perform correlation (default: Pearson) between all model patterns and the test patterns
% This is a quick and easy way to compute all the test items at once.

% 1. Build a matrix with [ModelA, ModelB,... ModelN, Test1, Test2,... TestN]
% 2. Run correlation over all of them, and get a the matrix of corr coeffs.
% 3. First column is Model A vs. all others, Second column is Model B vs.
%	all others, nth column is Model N vs. all others.
corr_matrix = atanh(corr([model_patterns,test_dat'],'type',opts.corr_stat,'rows','pairwise'));

% Isolate the columns representing the model_patterns, and the rows
% representing the test_data to get the correlations for each item
% in test data against all the model patterns.
test_model_corrs = corr_matrix(length(model_classes)+1:end,1:length(model_classes));

%% Save out the classification results based on greatest correlation coefficient for each test pattern
% Initialize empty cell matrix for classifications
classification = cell(size(test_dat,1),1);

if opts.exclusive && length(model_classes)==size(test_dat,1)
    
    %% Fill in the case for exclusive labels
    % wherein each class can only be assigned to one row of test data
    % (e.g., at Participant level when you are looking at
    % condition-averaged data, and only one pattern per condition)
    if size(test_model_corrs,1)==2
        if trace(test_model_corrs) > trace(rot90(test_model_corrs))
            classification(1) = model_classes(1);
            classification(2) = model_classes(2);
        elseif trace(test_model_corrs) < trace(rot90(test_model_corrs))
            classification(1) = model_classes(2);
            classification(2) = model_classes(1);
        else
            % If both options are equal, randomly assign the two labels to
            % the two observations.
            if ~isfield(opts,'tiebreak') || opts.tiebreak
                order = randperm(2); % returns [1 2] or [2 1] with equal probability
                classification(1) = model_classes(order(1));
                classification(2) = model_classes(order(2));
            else
                classification(1) = model_classes(1);
                classification(2) = model_classes(2);
            end
        end
    else
        % The search-all-label-permutations method would work here for 3 to
        % 10 classes, but the search space becomes too large after 10.
        disp('Currently no method for exclusive labeling with >2 test cases');
    end
    
else
    % If doing n-way classification (not all-possible-pairwise)
    if ~isfield(opts,'pairwise') || opts.pairwise==false
        
        % Adjust all values in test_model_corrs by <1% of the smallest
        % observed difference to prevent ties (randomly adjusts matched
        % values by tiny amount not relevant to classification).
        diffs = diff(sort(test_model_corrs(:)));
        min_diff = min(diffs(diffs>0));
        if isempty(min_diff), min_diff = min(test_model_corrs(:)); end
        test_model_corrs = test_model_corrs + rand(size(test_model_corrs))*min_diff/100;
        
        % Classify based on the maximum correlation
        [rating, test_class_idx] = max(test_model_corrs,[],2);
        classification = model_classes(test_class_idx);
        
    end
    % put the labels back in the order they were put in as
    [~,reorder_test] = sort(test_order);
    classification = classification(reorder_test);
end
