function new_names = get_names_from_orders(mcp_data, c, order, condition_names)
%% retrive the exemplar names for relabelling from category to exemplar

% mcp_data is the mcp data struct
% c is the old conditions
% order is the order the subject saw the labels in (from stimuli mat)
% condition_names are the new conditions


max_trials = sum(sum(mcp_data.fNIRS_Data.Onsets_Matrix));

new_names = {};
for x = 1:max_trials
    temp = regexp(order{x, 1}, '(1|2)', 'split');
    if strcmp(temp{1}, condition_names{c})
        new_names{end+1} = order{x};
    end      
end

end