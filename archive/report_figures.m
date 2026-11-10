% Run this AFTER diabetes_ga.m (it needs fis0, fis1 and the data in the workspace).
% Saves the report figures as PNG files in the same folder as the script.

out = fileparts(mfilename('fullpath'));   % the folder this script is saved in

figure; plotmf(fis0,'input',1); title('Glucose MFs - before GA');
exportgraphics(gcf, fullfile(out,'glucose_mf_before.png'), 'Resolution', 200);
figure; plotmf(fis1,'input',1); title('Glucose MFs - after GA');
exportgraphics(gcf, fullfile(out,'glucose_mf_after.png'), 'Resolution', 200);
figure; plotmf(fis0,'input',2); title('BMI MFs - before GA');
exportgraphics(gcf, fullfile(out,'bmi_mf_before.png'), 'Resolution', 200);
figure; plotmf(fis1,'input',2); title('BMI MFs - after GA');
exportgraphics(gcf, fullfile(out,'bmi_mf_after.png'), 'Resolution', 200);

% Control surface: Glucose and BMI against Risk
figure; gensurf(fis1, [1 2]); title('Control surface: Glucose, BMI -> Risk');
exportgraphics(gcf, fullfile(out,'control_surface.png'), 'Resolution', 200);

% Rule viewer (opens its own window; take a screenshot of it for the report)
ruleview(fis1);

fprintf('Saved PNG files to: %s\n', out);