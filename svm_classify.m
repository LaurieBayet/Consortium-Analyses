function [classification, comparisons] = svm_classify(train_data, train_labels, test_data, test_labels, opts)
% svm_classify implements support vector machine 
% input: model_data - data from the subjects in the group model
%        test_data -  labels for the subjects in the group model
%        test_data - subject data we want to predict for
%        test_labels - subject labels, unused
%        opts (optional) - an options struct

%% determine if we use pairwise comparisons 
if ~exist('opts','var') || ~isfield(opts, 'pairwise') || isempty(opts)
    pairwise = false;
else
    pairwise = opts.pairwise;
    opts = rmfield(opts,'pairwise');
end

if ~exist('pairwise','var')
    pairwise = false;
end

%% parse out the classification parameters
input = parse_opts(opts);

%% if pairwise classification
if pairwise
    
    number_classes = length(unique(train_labels));
    class_names = unique(train_labels);
    class_count = sum(strcmp(test_labels, class_names{1}));
    list_of_comparisons = combnk([1:number_classes],2);
    number_of_comparisons = size(list_of_comparisons,1);
    results_of_comparisons = cell((class_count*2), 2, number_of_comparisons);
    
    for this_comp = 1:number_of_comparisons
        
        % figure out what classes we need to keep for this round of
        % comparisons
        test_classes = list_of_comparisons(this_comp,:);
        test_class_names = class_names(test_classes);
        
        % select just the rows of the training data and labels that we will
        % need for this comparison
        train_dat = [train_data(strcmp(train_labels,test_class_names(1)),:);...
                     train_data(strcmp(train_labels,test_class_names(2)),:)];
        train_labs = [train_labels(strcmp(train_labels,test_class_names(1)));...
                      train_labels(strcmp(train_labels,test_class_names(2)))];
        
        
        % select just the rows of the test data and labels that we will
        % need for this comparison
        test_dat = [test_data(strcmp(test_labels,test_class_names(1)),:);...
                    test_data(strcmp(test_labels,test_class_names(2)),:)];
        test_labs = [test_labels(strcmp(test_labels,test_class_names(1)));...
                     test_labels(strcmp(test_labels,test_class_names(2)))];
        
        t = templateSVM(input{:});
        svm_model = fitcecoc(train_dat, train_labs,'Learners', t);

        %svm_model = fitcsvm(train_dat, train_labs, input{:});
        
        classification = predict(svm_model, test_dat);
        
        results_of_comparisons(:,1, this_comp) = classification;
        results_of_comparisons(:,2, this_comp) = test_labs;  
    end
    comparisons = list_of_comparisons;
else
    t = templateSVM(input{:});
    svm_model = fitcecoc(train_data, train_labels, 'Learners', t);
    classification = predict(svm_model, test_data);
    comparisons = test_labels;
end

end
