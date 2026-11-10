clear; clc;

%% ---------- Load and clean data ----------
here = fileparts(mfilename('fullpath'));
data = readtable(fullfile(here, 'pima-indians-diabetes.csv'), 'ReadVariableNames', false);
data.Properties.VariableNames = {'Pregnancies','Glucose','BloodPressure', ...
    'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'};

% zeros in these columns mean "missing", so replace with the median
for v = {'Glucose','BloodPressure','BMI'}
    col = data.(v{1});
    col(col==0) = median(col(col~=0));
    data.(v{1}) = col;
end

X = data{:, {'Glucose','BMI','Age','BloodPressure'}};
y = data.Outcome;

disp(head(data))
fprintf('Rows: %d, Diabetic: %.1f%%\n', height(data), mean(y)*100);

%% ---------- Build Mamdani FIS ----------
fis = mamfis('Name','DiabetesRisk');

fis = addInput(fis,[40 200],'Name','Glucose');
fis = addMF(fis,'Glucose','trapmf',[40 40 90 110],'Name','Low');
fis = addMF(fis,'Glucose','trimf',[100 125 150],'Name','Medium');
fis = addMF(fis,'Glucose','trapmf',[140 165 200 200],'Name','High');

fis = addInput(fis,[15 70],'Name','BMI');
fis = addMF(fis,'BMI','trapmf',[15 15 23 27],'Name','Normal');
fis = addMF(fis,'BMI','trimf',[25 30 35],'Name','Overweight');
fis = addMF(fis,'BMI','trapmf',[32 38 70 70],'Name','Obese');

fis = addInput(fis,[20 85],'Name','Age');
fis = addMF(fis,'Age','trapmf',[20 20 30 40],'Name','Young');
fis = addMF(fis,'Age','trimf',[35 48 60],'Name','Middle');
fis = addMF(fis,'Age','trapmf',[55 65 85 85],'Name','Old');

fis = addInput(fis,[20 125],'Name','BloodPressure');
fis = addMF(fis,'BloodPressure','trapmf',[20 20 65 80],'Name','Low');
fis = addMF(fis,'BloodPressure','trimf',[70 80 90],'Name','Normal');
fis = addMF(fis,'BloodPressure','trapmf',[85 100 125 125],'Name','High');

fis = addOutput(fis,[0 100],'Name','Risk');
fis = addMF(fis,'Risk','trimf',[0 0 40],'Name','Low');
fis = addMF(fis,'Risk','trimf',[25 45 65],'Name','Medium');
fis = addMF(fis,'Risk','trimf',[60 100 100],'Name','High');

%% ---------- Rules ----------
rules = [
 % --- High glucose ---
 "Glucose==High => Risk=High"
 "Glucose==High & BMI==Obese => Risk=High"
 "Glucose==High & BMI==Overweight => Risk=High"
 "Glucose==High & Age==Old => Risk=High"
 "Glucose==High & Age==Middle => Risk=High"
 "Glucose==High & BloodPressure==High => Risk=High"
 % --- Medium glucose ---
 "Glucose==Medium & BMI==Obese => Risk=High"
 "Glucose==Medium & BMI==Overweight => Risk=Medium"
 "Glucose==Medium & BMI==Normal => Risk=Low"
 "Glucose==Medium & Age==Old => Risk=Medium"
 "Glucose==Medium & Age==Middle => Risk=Medium"
 "Glucose==Medium & Age==Young => Risk=Low"
 "Glucose==Medium & BloodPressure==High => Risk=Medium"
 % --- Low glucose ---
 "Glucose==Low & BMI==Normal => Risk=Low"
 "Glucose==Low & BMI==Overweight => Risk=Low"
 "Glucose==Low & BMI==Obese => Risk=Medium"
 "Glucose==Low & Age==Young => Risk=Low"
 "Glucose==Low & Age==Old => Risk=Medium"
 "Glucose==Low & BloodPressure==Normal => Risk=Low"
];
fis = addRule(fis, rules);
fprintf('Number of rules in FIS: %d\n', numel(fis.Rules));   % should print 19

%% ---------- Evaluate and sweep threshold ----------
score = evalfis(fis, X);

thr = 30:1:95;
accs = zeros(size(thr));
for i = 1:numel(thr)
    accs(i) = mean((score > thr(i)) == y) * 100;
end
[bestAcc, idx] = max(accs);
bestThr = thr(idx);
fprintf('Best accuracy: %.2f%% at threshold %d\n', bestAcc, bestThr);

figure; plot(thr, accs, '-o'); grid on;
xlabel('Threshold'); ylabel('Accuracy (%)'); title('Threshold sweep');

% Which patients sit at exactly 50 (Medium-only, or no rule fired)?
at50 = find(abs(score-50) < 1e-6);
fprintf('Patients at exactly 50: %d\n', numel(at50));
if ~isempty(at50)
    disp(X(at50(1:min(10,end)), :));      % Glucose, BMI, Age, BloodPressure
end