function allsubj_results = create_results_struct(within_subject, summed_mcpa, parsed_input, all_sets, num_subj, num_sets, num_feature, num_cond, final_dimensions)
%% create a struct to store classification results. 
% This will be later used in permutation testing 

allsubj_results = []; % create empty structs
allsubj_results.created = datestr(now);
allsubj_results.summed_mcpa_patterns = summed_mcpa.patterns;
allsubj_results.summarize_dimensions = summed_mcpa.summarize_dimensions;
allsubj_results.subsets = all_sets;
allsubj_results.test_handle = parsed_input.Results.test_handle;
allsubj_results.incl_channels = parsed_input.Results.incl_channels;
allsubj_results.conditions = parsed_input.Results.conditions;
allsubj_results.incl_subjects = parsed_input.Results.incl_subjects;
allsubj_results.verbose = parsed_input.Results.verbose;
allsubj_results.max_sets = parsed_input.Results.max_sets;
allsubj_results.final_dimensions = final_dimensions;
allsubj_results.event_types = summed_mcpa.event_types;
allsubj_results.dimensions = summed_mcpa.dimensions;
allsubj_results.opts_struct = parsed_input.Results.opts_struct;
allsubj_results.oxy_or_deoxy = parsed_input.Results.oxy_or_deoxy; 

%% get max sessions completed for later accuracy struct
num_dims = ndims(summed_mcpa.patterns);
session_dim = num_dims -1;
allsubj_results.max_sessions = size(summed_mcpa.patterns, session_dim);

%% create accuracy struct
for cond_id = 1:num_cond % now create place holders for decoding accuracies 
    allsubj_results.accuracy(cond_id).condition = allsubj_results.conditions(cond_id);
    allsubj_results.accuracy(cond_id).subjXfeature = nan(num_subj,num_feature);
    allsubj_results.accuracy(cond_id).subsetXsubj = nan(num_sets,num_subj);
    if within_subject
         allsubj_results.accuracy(cond_id).subjXsession = nan(num_subj,allsubj_results.max_sessions);
    end
end

end
