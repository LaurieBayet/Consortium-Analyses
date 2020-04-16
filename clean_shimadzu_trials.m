function new_MCP_struct = clean_shimadzu_trials(MCP_struct)
%% This function takes in an MCP struct and cleans trials that were double recorded by the Shimadzu machine
% also removes any trial after trial 240
% input: an MCP struct
% output: an MCPstruct without excess trials

for i = 1:length(MCP_struct) % for each participant
    stims=(MCP_struct(i).fNIRS_Data.Onsets_Matrix * [1;2;3;4;5;6;7;8]); % get trials types
    locs = find(stims); % get when those trial types are happening
    l = {};
    trial_type = {};
    
    
    %% first we look for repeated trials
    for j = 2:length(locs) % for each stimulus location
        curr_elem = locs(j); % get the current element
        past_elem = locs(j-1); % get the previous element
        
        if stims(curr_elem) == stims(past_elem) % are those elements the same
            l{end+1} = past_elem; % store them if yes
            trial_type{end+1} = stims(past_elem);
        end
    end
    

    %% Then remove repeaed trials
    for k = 1:length(trial_type) % remove repeated elements
        MCP_struct(i).fNIRS_Data.Onsets_Matrix(l{k},trial_type{k}) = 0;
    end
    
    %% Then remove trials over 240
    leave_extra = 240 + length(trial_type); % remove excess trials past 240
    if length(locs) > leave_extra
        to_remove = locs(leave_extra+1:end);
        MCP_struct(i).fNIRS_Data.Onsets_Matrix(to_remove,:) = 0; 
    end
       
end

new_MCP_struct = MCP_struct;

end


