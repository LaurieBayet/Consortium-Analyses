function [group_dat, subject_dat] = standardize_data(group, subject)
%% standardize data so that we are working with z-scores
% Arguments:
% group: group model data (training data for classification)
% subject: left out subject data (testing data for classification)

average = mean(group);
sd = std(group);
group_dat = (group-average) ./ sd;

if ~isstruct(subject)
    subject_dat = (subject - average) ./ sd;
else
    subject_dat = nan;
end

end