function [classification, comparisons] = rsa_classify(model_data, model_labels, test_data, test_labels, opts)
%% rsa_classify implements a correlation-based, similarity-space classifier
% following Zinszer, Bayet, Emberson, Raizada, & Aslin's (2018, Neurophotonics)
% method for all-possible-pairwise-comparisons and Zinszer, Anderson, Kang,
% Wheatley, & Raizada's (2016) approach for n-way comparison
%
% User have the option to create similarity matrices or dissimilarity matrices.
% If only the metric to create RSA matrices is given, it will default to similarity space
% if the metric is pearson, spearman, or kendall. Note: pearson and spearman are also 
% distance functions, so they could also be used to create dissimilarity matrices. 
% If creating dissimilarity matrices with these distance functions, please be sure to set
% opts.similarity_space to false. 
% 
% Users can choose what metric to create matrices with by changing the value of
% opts.metric. Users can choose to create similarity matrices by setting 
% opts.similarity_space to true.
% 
% If similarity matrices are created, they will be be adjusted with the hyperbolic arctangent.
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
end
if ~isfield(opts, 'similarity_space')
    if isfield(opts, 'metric')
        switch opts.metric
            case {'spearman', 'pearson', 'kendall'}
                warning('Similarity or Dissimilarity space has not been defined. Based on your chosen metric, we will put the data in similarity space.')
                opts.similarity_space = true;
            otherwise
                warning('Similarity or Dissimilarity space has not been defined. Based on your chosen metric, we will put the data in dissimilarity space.')
                opts.similarity_space = false;
        end
    else
        warning('Similarity or Dissimilarity space has not been defined. We will put the data in similarity space with the Spearman correlation.')
        opts.similarity_space = true;
    end
end
if ~isfield(opts, 'metric')
    if opts.similarity_space
        opts.metric = 'spearman';
    else
        opts.metric = 'euclidean';
    end
end
if ~isfield(opts, 'exclusive')
    opts.exclusive = true;
end
if ~isfield(opts, 'pairwise')
    opts.pairwise = false;
end
if ~isfield(opts, 'tiebreak')
    opts.tiebreak = true;
end
if ~isfield(opts, 'verbose')
    opts.verbose = 0;
end

%% Pull a list of all the unique classes / conditions, preserving order
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

%% remove where we have all NaNs

empty_x_vals_model = [];
empty_y_vals_model = [];
for i = 1:size(model_data,4)
    for j = 1:size(model_data,3)
        [x,y] = find(isnan(model_dat(:,:,j,i)));
        if length(unique(x)) == size(model_dat,1) && length(unique(y)) == size(model_dat,2)
            continue;
        else
            empty_x_vals_model = [empty_x_vals_model; x];
            empty_y_vals_model = [empty_y_vals_model; y];
        end
    end      
end
empty_x_vals_model = unique(empty_x_vals_model);
empty_y_vals_model = unique(empty_y_vals_model);


empty_x_vals_test = [];
empty_y_vals_test = [];
for i = 1:size(test_data,4)
    for j = 1:size(test_data,3)
        [x,y] = find(isnan(test_data(:,:,j,i)));
        
        
        if length(unique(x)) == size(test_data,1) && length(unique(y)) == size(test_data,2)
            continue;
        else
            empty_x_vals_test = [empty_x_vals_test; x];
            empty_y_vals_test = [empty_y_vals_test; y];
        end
    end      
end

empty_x_vals_test = unique(empty_x_vals_test);
empty_y_vals_test = unique(empty_y_vals_test);

cols = 1:size(model_data,2);
rows = 1:size(model_data,1);

if size(model_data,2) > 8
    if length(empty_x_vals_model) == size(model_data,1) && length(empty_y_vals_model) ~= size(model_data,2)
        % remove columns
        remove_cols = union(empty_y_vals_model, empty_y_vals_test);
        keep = ~ismember(cols, remove_cols);
        model_dat = model_dat(:,keep', :,:);
        test_dat = test_dat(:, keep', :,:);   
    elseif length(empty_x_vals_model) == size(model_data,1) && length(empty_y_vals_model) == size(model_data,2)
        % this is an empty session
        warning('There is no data in this session')
    end
else
    if length(empty_x_vals_test) == size(model_data,1) && length(empty_y_vals_test) ~= size(test_data,2)
        test_cols = 1:size(test_dat,2);
        remove_cols = union(empty_y_vals_model, empty_y_vals_test);
        keep = ~ismember(test_cols, remove_cols);
        test_dat = test_dat(:, keep', :,:);
    end 
end
    
    


%% Build similarity structures
% Transform the model_data into similiarty or dissimilarity structures for each session by
% correlating between conditions or finding pairwise differences between conditions and then 
% average the similarity/dissimilarity structures together into a single model.

if opts.similarity_space % create similarity structures
    %  iterate through all the layers (3rd dimension) and create
    % correlation matrices 
    
    if ndims(model_dat) > 2
        model_mat = nan(size(model_dat,1),size(model_dat,1),size(model_dat,3),size(model_dat,4));
        for i = 1: (size(model_dat,3)*size(model_dat,4))
            model_mat(:,:,i) = corr(model_dat(:,:,i)', 'type', opts.metric);
        end

        % then average across each participants' sessions
        training_matrix = nanmean(model_mat,3);
        % then average across participants
        training_matrix = nanmean(training_matrix,4);
        training_matrix = atanh(training_matrix);
    else
        training_matrix = model_dat;
    end
    try
        % then repeat to create test data
        test_mat = nan(size(test_dat,1),size(test_dat,1),size(test_dat,3),size(test_dat,4));
        for i = 1: (size(test_dat,3)*size(test_dat,4))
            test_mat(:,:,i) = corr(test_dat(:,:,i)', 'type', opts.metric);
        end
        test_matrix = nanmean(test_mat,3);
        test_matrix = nanmean(test_matrix,4);

        test_matrix = atanh(test_matrix);
    catch
        model
    end
   
else % create dissimilarity structures
    
    if ndims(model_data) > 2
        % loop through each participants' sessions to find pairwise differences
        model_mat = nan(size(model_dat,1),size(model_dat,1),size(model_dat,3),size(model_dat,4));
        for i = 1: (size(model_dat,3)*size(model_dat,4))
            model_mat(:,:,i) = squareform(pdist(model_dat(:,:,i), opts.metric));
        end
        % then average across each participants' sessions
        training_matrix = nanmean(model_mat,3);
        % then average across participants
        training_matrix = nanmean(training_matrix,4);
    else
        training_matrix = model_dat;
    end
    
    % then repeat to create test data
    test_mat = nan(size(test_dat,1),size(test_dat,1),size(test_dat,3),size(test_dat,4));
    for i = 1: (size(test_dat,3)*size(test_dat,4))
        test_mat(:,:,i) = squareform(pdist(test_dat(:,:,i), opts.metric));
    end
    test_matrix = nanmean(test_mat,3);
    test_matrix = nanmean(test_matrix,4);
end


%% Visualize the matrices

if opts.verbose > 1
    
    plot_idx = 1;
    figure
    for session_idx = 1:size(model_data,3)
        for subject_idx = 1:size(model_data,4)
            subplot(size(model_data,3),size(model_data,4),plot_idx);
            imagesc(model_mat(:,:,session_idx, subject_idx))
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
            imagesc(test_mat(:,:,session_idx, subject_idx))
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
if sum(isnan(test_matrix(:)))==(numel(test_matrix)-8) || sum(isnan(test_matrix(:)))==numel(test_matrix) || sum(isnan(training_matrix(:)))==numel(training_matrix)
     warning('One or both input matrices contains all NaN values. this one triggered');
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
        results_of_comparisons(perm_idx) = corr(train_vec,tmp_test_vec,'rows','pairwise');
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
    
    [accuracy, comparisons] = pairwise_rsa_test(test_matrix,training_matrix);
    
    classification = comparisons;
    
    for acc = 1:length(accuracy)
        if isnan(accuracy(acc))
            classification(acc,:) = nan;
        else
            if ~accuracy(acc)
                classification(acc,:) = classification(acc,end:-1:1);
            end
        end
    end
   
    classification = num2cell(classification);
    for class = 1:size(classification,1)
        if ~any(isnan([classification{class,:}]))
            classification(class,:) = model_classes([classification{class,:}]);
        end
    end
    
    comparisons = model_classes(comparisons);
    
    results_of_comparisons = cell(size(classification,2), 2, size(classification,1));
        for comp = 1:size(classification,1)
            results_of_comparisons(:,1,comp) = classification(comp,:)';
            results_of_comparisons(:,2,comp) = comparisons(comp,:)';
        end
        

    classification = results_of_comparisons;
    
    
end


end
