function summarized_MCPA_struct = summarize_MCPA_Struct(summary_function,MCPA_struct, summarize_dimensions)
%SUMMARIZE_MCPA_STRUCT Convert an MCPA struct of windowed data to
%multivariate patterns using any summarizing function, such as nanmean.


%% Convert the inputs to correct format
% Convert summary function into a function handle
if ischar(summary_function)
    summary_function = str2func(summary_function);
elseif iscell(summary_function)
    summary_function = str2func(summary_function{:});
elseif isempty(summary_function)
    warning('No summarizing function specified. Reshaping dimensions only.')
end

% Convert summarize_dimensions into cell array
if ischar(summarize_dimensions)
    warning('Please provided cell array of dimensions to summarize. Automatically converting...')
    summarize_dimensions = {summarize_dimensions};
elseif isnumeric(summarize_dimensions)
    warning('Please provided cell array of dimensions to summarize. Looking up dimension names in MCPA''s dimensions field...')
    if ~isfield(MCPA_struct,'dimensions')
        warning('No ''dimension'' field found in MCPA struct, so no dimension labels available. Assigning numeric labels.')
        MCPA_struct.dimensions = cellfun(@(x) strcat('PatternDim',x), cellfun(@num2str, num2cell(1:ndims(MCPA_struct.patterns)),'UniformOutput',false),'UniformOutput',false);
    end
    summarize_dimensions = MCPA_struct.dimensions(summarize_dimensions);
    dim_str = join(summarize_dimensions,', ');
    fprintf('Converted numeric dimension indices to: %s', dim_str{:})
end

%% Create copies of dimensions and of pattern matrix
dimension_labels = MCPA_struct.dimensions;
pattern_matrix = MCPA_struct.patterns;

%% Get summarized
% Step through the dimensions listed from left to right. If a dimension
% contains the concatenation operator (X), then reshape the pattern matrix
% and rename the dimension(s). Otherwise, perform the summarizing function
% on the dimension.

%% new approach
for curr_dim = 1:length(summarize_dimensions)
    
    % Perform concatenations
    concat = strsplit(summarize_dimensions{curr_dim}, 'X');
    if length(concat) == 2
        try
            first_dim_to_concat = find(strcmp(dimension_labels,concat{1}));
            second_dim_to_concat = find(strcmp(dimension_labels,concat{2}));
            pattern_matrix = concatenate_dimensions(pattern_matrix, [first_dim_to_concat,second_dim_to_concat]);
            
            % get the new dimensions
            dimension_labels{first_dim_to_concat} = [dimension_labels{first_dim_to_concat}, '+', dimension_labels{second_dim_to_concat}];
            dimension_labels{second_dim_to_concat} = [];
            
        catch concat_error
            warning(concat_error.message)
            error('Failed to concatenate dimensions: %s', summarize_dimensions{curr_dim})
        end
    elseif ~isempty(summary_function)
        try
            % Perform summary function over the specified dimension
            dim_to_summarize = find(strcmp(dimension_labels, summarize_dimensions{curr_dim}));
            pattern_matrix = summary_function(pattern_matrix,dim_to_summarize);
            
            % Then keep track of remaining dimensions in list
            dimension_labels{dim_to_summarize} = [];
        catch averaging_error
            warning(averaging_error.message);
            error('Failed to successfully summarize dimension: %s', summarize_dimensions{curr_dim});
        end
    end
end

if ~isempty(strcmp('session', dimension_labels))
    session_idx = find(strcmp('session', dimension_labels));
    s = size(pattern_matrix);
    if s(session_idx) == 1 % we don't want to loose this dimension if it is 1
        to_remove = s ~= 1;
        to_remove(session_idx) = 1;
        summarized_MCPA_struct_pattern = reshape(pattern_matrix,s(to_remove));
    else
        summarized_MCPA_struct_pattern = squeeze(pattern_matrix);
    end
end    

dimension_labels = dimension_labels(~cellfun('isempty',dimension_labels));

%% Return the new struct
try
    summarized_MCPA_struct = MCPA_struct;
    summarized_MCPA_struct.created = datestr(now);
    summarized_MCPA_struct.summarizing_function = summary_function;
    summarized_MCPA_struct.patterns = summarized_MCPA_struct_pattern;
    summarized_MCPA_struct.dimensions = dimension_labels;
    summarized_MCPA_struct.summarize_dimensions = summarize_dimensions;
    fprintf(' Done.\n');
    
catch
    error('Failed to create the new struct');
end



end
