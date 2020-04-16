function [train_data, train_labels, test_data, test_labels] = split_test_and_train(fold_idx, conditions, pattern_data, event_types, final_dimensions, dimension_labels, test_events, train_events) 


%% step 1: find the locations of the conditions we would like to classify
% only need them if we aren't supplying test and train events
if isempty(test_events)
    % Set logical flags for indexing the conditions that will be compared.
    % Loop through the whole list of conditions and create flags for each.
    test_cond_flags = cell(length(conditions),1); % These are, for the moment, empty
    train_cond_flags = cell(length(conditions),1);
    
    for cond_idx = 1:length(conditions)
        if ischar(conditions{cond_idx}) || isstring(conditions{cond_idx}) || iscellstr(conditions{cond_idx})
            [~, ~, test_cond_flags{cond_idx}] = intersect(conditions{cond_idx},event_types);
            [~, ~, train_cond_flags{cond_idx}] = intersect(conditions{cond_idx},event_types);
        else
            test_cond_flags{cond_idx} = conditions{cond_idx};
            train_cond_flags{cond_idx} = conditions{cond_idx};
        end
    end
    test_cond_flags = [test_cond_flags{:}]';
    train_cond_flags = [train_cond_flags{:}]';
else
    test_cond_flags = test_events;
    train_cond_flags = train_events;   
end

%% Do we need to concatenate any dimensions to get to the final form for classification
% and if so where are the dimensions that need concatenation
split_dims = cellfun(@(s) strsplit(s,'X'), final_dimensions, 'UniformOutput', false);
has_dims_to_concat = cellfun('size', split_dims,2) > 1;

if any(has_dims_to_concat)
    concat_dims = split_dims{has_dims_to_concat};
    to_concat = [];
    for i = 1:length(concat_dims)
        idx = find(strcmp(dimension_labels, concat_dims{i}));
        to_concat = [to_concat, idx];
    end    
end

%% separate out which participants or sessions go in the training data
group_vec = [1:size(pattern_data,ndims(pattern_data))];
group_vec = group_vec(group_vec~=fold_idx);

%% step 3: get the train data

% subset out train
if ndims(pattern_data) == 3
    temp_train = pattern_data(train_cond_flags,:,group_vec);
else
    temp_train = pattern_data(train_cond_flags,:,:,group_vec);
end

% then concatenate train
if any(has_dims_to_concat)
    train_data = concatenate_dimensions(temp_train, to_concat);
else
    train_data = temp_train;
end

%% step 2: get test data

% subset out test
if ndims(pattern_data) == 3
    temp_test = pattern_data(test_cond_flags,:,fold_idx);
else
    temp_test = pattern_data(test_cond_flags,:,:,fold_idx);
end

% then concatenate test
if any(has_dims_to_concat)
    % need to account for the assignment of 1 fold dropping the last dimension
    [max_value, max_ind] = max(to_concat);
    if max_value > ndims(temp_test)
        to_concat(max_ind) = [];
    end

    % create the test set
    test_data = concatenate_dimensions(temp_test, to_concat);
else
    test_data = temp_test;
end

%% step 4: get the class labels for test and train data

% first need to know how many times each label is represented in the data
train_label_repetitions = round(size(train_data, 1)/(length(unique(event_types))*(length(event_types(train_cond_flags))/length(unique(event_types(train_cond_flags))))));
test_label_repetitions = round(size(test_data, 1)/(length(unique(event_types))*(length(event_types(test_cond_flags))/length(unique(event_types(test_cond_flags))))));


% then create vector of labels
train_labels = repmat(event_types(train_cond_flags)', 1,train_label_repetitions)';
test_labels = repmat(event_types(test_cond_flags)', 1,test_label_repetitions)';
  
    
end
