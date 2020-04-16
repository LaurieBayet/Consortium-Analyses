function p = parse_inputs(MCP_struct, varargin)
%% input parser 
% takes input from function that calls it and creates a struct to store
% clasification parameters
% if not input is provided for a particular variable, it creates default values

% Arguments:
% MCP_struct: MCP data structure
% varargin: input to main function

p = inputParser;

% parameters used for all kinds of classifiers
addParameter(p,'incl_channels',[1:max(arrayfun(@(x) size(x.Experiment.Runs(1).Transformation_Matrix,1),MCP_struct))],@isnumeric);
addParameter(p,'incl_subjects',[1:length(MCP_struct)],@isnumeric);
addParameter(p,'incl_sessions',[1:max(arrayfun(@(x) length(x.Experiment.Runs),MCP_struct))],@isnumeric);
addParameter(p,'time_window',[2,6],@isnumeric);
addParameter(p,'baseline_window',[-5 0],@isnumeric);
addParameter(p,'conditions',unique(cellstr(char(cellfun(@(x) char(x{:}), arrayfun(@(x) unique({x.Experiment.Conditions.Name},'stable'),MCP_struct, 'UniformOutput',false),'UniformOutput',false))),'stable'),@iscell);
addParameter(p,'summary_handle',@nanmean);
addParameter(p,'setsize',max(arrayfun(@(x) size(x.Experiment.Runs(1).Transformation_Matrix,2),MCP_struct)),@isnumeric);
addParameter(p,'max_sets',1000000,@isnumeric);
addParameter(p,'test_handle',@mcpa_classify);
addParameter(p,'opts_struct',[],@isstruct);
addParameter(p,'norm_data', false, @islogical);
addParameter(p,'verbose',true,@islogical);
addParameter(p, 'summarize_dimensions', {});
addParameter(p, 'final_dimensions', {});
addParameter(p, 'oxy_or_deoxy', 'Oxy', @ischar);

% for within subjects decoding with 1 session
addParameter(p, 'test_percent', []);

% parameters for working in different feature spaces
addParameter(p, 'feature_space', 'channel_space', @ischar);
addParameter(p, 'incl_features', [1:max(arrayfun(@(x) size(x.Experiment.Runs(1).Transformation_Matrix,2),MCP_struct))],@isnumeric); 
 
% parameters used if norming the data
addParameter(p,'norm_withinSessions', true, @islogical);
addParameter(p, 'norm_function', @minMax_scale);
addParameter(p, 'minMax', [0,1], @isnumeric);

% parameters only used in nfold_generalize_ParticipantLevel
addParameter(p,'cond_key',{});
addParameter(p,'test_marks',{});

parse(p,varargin{:});

end
