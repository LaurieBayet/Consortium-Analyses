function [event_matrix] = MCP_get_subject_events(mcp_struct, features, channels, time_window, event_types, base_window, oxy_or_deoxy, session_index)

%MCP_GET_SUBJECT_EVENTS Returns a matrix that contains HbO data for target 
%subject in each type, feature, time, and type repetition.
%
% First we construct the index matrix that contains specific index of event
% onsets in the hemoglobin data. Then use the index matrix to find 
% corresponding windows of hemoglobin data and copy them to the output.
%
% Output:
% event_matrix: time x feature x rep x condition
%
% Chengyu Deng & Benjamin Zinszer 5 may 2017
% revised bdz 29 aug 2019

%% If no base-lining window (base_window) is specified, mark NaN flag
if ~exist('base_window','var')
    base_window = NaN;
end

%% MCP_get_subject_events will run recursively for multiple subjects
% If the mcp_struct contains multiple elements (i.e., a struct for each
% subject), then the function will call itself and re-evaluate on the
% contents of each struct, returning event_matrix in a separate cell for
% each element in the struct (i.e., a cell for each subject).

% is this a problem for the transformation matrix???
if length(mcp_struct)>1
    event_matrix = cell(length(mcp_struct),1);
    for subj_num = 1:length(mcp_struct)
        event_matrix{subj_num} = MCP_get_subject_events(mcp_struct(subj_num), features, channels, time_window, event_types, base_window, oxy_or_deoxy, session_index);
    end
    return
end

%% Index matrix handling
% find the location of hemoglobin data for this session
session_locs = mcp_struct.Experiment.Runs(session_index).Index';
% Extract hemoglobin data and marks from the MCP struct
hemo_timeser = mcp_struct.fNIRS_Data.Hb_data.(oxy_or_deoxy)(session_locs, channels);

% Extract the transformation matrix and if its converting to ROI space,
% weight it
if size(mcp_struct.Experiment.Runs(session_index).Transformation_Matrix,1) == size(mcp_struct.Experiment.Runs(session_index).Transformation_Matrix,2)
    % if we're using the identity matix, then we can just take the original transformation matrix
    transformation_mat = mcp_struct.Experiment.Runs(session_index).Transformation_Matrix(channels, channels);
else
    % if its converting into Brodmann's areas, then we first need to
    % extract the transformation matrix
    transformation_mat = mcp_struct.Experiment.Runs(session_index).Transformation_Matrix(channels,features);
    % then we need to weight it so that each area sums to 1
    weight = sum(transformation_mat,1);
    transformation_mat = transformation_mat ./ weight;
end

% transform the hemodynamic timeseries
hemo_timeser = hemo_timeser * transformation_mat;

marks_vec = mcp_struct.fNIRS_Data.Onsets_Matrix(session_locs,:);

% Handle different type of marks vector
if size(marks_vec, 2) > 1
    
    % Determine the maximum number of reps for any given marker (iterates
    % through the columns of marks_vec and finds values >0, aka events)
    max_condition_type = max(sum(marks_vec>0,1));
    
    marks_mat = nan(max_condition_type, size(marks_vec,2));
    
    for type_i = 1:size(marks_vec,2)
        %Find the array index in the marks_vec
        temp_marks = find(marks_vec(:, type_i) == 1); 
        marks_mat(1:length(temp_marks), type_i) = temp_marks;
    end
    
    % Remove rows whose entries are all NaN
    marks_mat = marks_mat(sum(isnan(marks_mat),2)<size(marks_mat,2),:);
    
elseif size(marks_vec, 2) == 1
    
    % marks_mat matrix's row is the max number of the condition in whole
    % conditions, and the column is the event types.
    marks_mat = nan(max(hist(marks_vec(marks_vec~=0))),length(event_types));
    
    for type_i = 1 : length(event_types)
        %Find the array index (vector index) in the marks_vec
        temp_marks = find(marks_vec == event_types(type_i));
        marks_mat(1:length(temp_marks), type_i) = temp_marks;
    end
    
    % Remove rows whose entries are all NaN
    marks_mat = marks_mat(sum(isnan(marks_mat),2)<size(marks_mat,2),:);
    
else
    %We shall never reach this part hopefully.
    fprintf('Unidentified data format. Fail to return right event matrix.');
    event_matrix = [];
    return
end

%% Extract the individual events for each event type
Fs_val = mcp_struct.fNIRS_Data.Sampling_frequency;
time_window_samp = round(time_window.*Fs_val); % converting time (s) to number of samples
base_window_samp = round(base_window.*Fs_val); % converting time (s) to number of samples

% The output matrix setup(time x features x type repetition x types)
num_samps = max(time_window_samp) - min(time_window_samp) + 1;
event_matrix = nan(num_samps, size(hemo_timeser,2), size(marks_mat, 1), length(event_types));
%%
for type_i = 1 : length(event_types)
    
    matched_conditions = cell2mat(arrayfun(@(x) strcmp(event_types{type_i},x.Name), mcp_struct.Experiment.Conditions,'UniformOutput',false)); 
    event_marks = [mcp_struct.Experiment.Conditions(matched_conditions).Mark];
    
    % Each instance of a given condition (i.e., each individual event) is
    % extracted and entered into the event_matrix. Since there are often an
    % unbalanced number of events per condition, the events_matrix is
    % preallocated to accomodate the largest number of events for any given
    % condition, and the conditions with fewer events have NaN values
    % filling in the rest. These NaNs are important to remember if you're
    % taking a mean value later (use mean(x,dim,'omitnan') or nanmean(x,dim)
    for event_j = 1 : length(marks_mat(:, event_marks))
        
        % Find the earliest time-point (in samples) that will be extracted
        % and the latest time-point (in samples) that will be extracted so
        % that we can determine the window of data to collect.
        earliest_samp = min( [time_window_samp(:); base_window_samp(:)] );
        latest_samp = max( [time_window_samp(:); base_window_samp(:)] );
        
        % For events where the full duration of the baseline+event is 
        % available, extract the event_data (within the time_window) and
        % rebaseline_data (within the base_window). rebaseline_data are
        % then averaged, and the average is removed from the event_data
        if marks_mat(event_j,event_marks) + earliest_samp > 0 && ...
            marks_mat(event_j,event_marks) + latest_samp <= length(hemo_timeser)
            
            % Grab the trial data (for this event) directly from the hemo
            % timeseries. It will be re-baselined momentarily.
            event_data = hemo_timeser(...
                (marks_mat(event_j,event_marks)+min(time_window_samp)) : ... % beginning of window
                (marks_mat(event_j,event_marks)+max(time_window_samp)) ...   % end of window
                ,:);
            
            % Get the hemo data from one scan prior to stimulus onset and
            % use that as the baselining value.
            if ~isnan(base_window)
                rebaseline_data = hemo_timeser(...
                    (marks_mat(event_j,event_marks)+min(base_window_samp)) : ...	% beginning of window
                    (marks_mat(event_j,event_marks)+max(base_window_samp)) ...   % end of window
                    ,:);
            else
                rebaseline_data = 0;
            end
            event_matrix(:,:,event_j,type_i) = event_data - nanmean(rebaseline_data);

        % For events that might get truncated (start or end of recording)
        else
            % For the moment, we carelessly discard these trials. There is
            % probably a better way to do it, but you have to match up the 
            % lengths of the vectors and figure out rebaselining. Todo.
            event_matrix(:,:,event_j:end,type_i) = NaN;
        end
    end

end

end
