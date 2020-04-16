function scaled_full_set = minMax_scale(full_dataset, input_struct)
%% This function scales data to a set minimum and maximum
% Arguments: 
% full_dataset: Either input a full Oxy or Deoxy dataset from one participant if you
% want to scale individual subject data or enter this as [] if you want to
% scale group level data
% testing_data: Either input the testing data (after we split test and
% train data) or input [] if we are scaling individual level data
% training_data: Input training data (after we split test and train data)
% or input [] if scaling individual level data
% input_struct: stucture we entered into folding function with our
% parameters

if isfield(input_struct, 'minMax')
    minimum = input_struct.minMax(1); % what minimum are we scaling to
    maximum = input_struct.minMax(2); % what maximum are we scaling to
else
    minimum = 0;
    maximum = 1;
end


mins = min(full_dataset);
maxes = max(full_dataset);
scaled_full_set = ((maximum - minimum)*((full_dataset - mins) ./ (maxes - mins))) + minimum;


end