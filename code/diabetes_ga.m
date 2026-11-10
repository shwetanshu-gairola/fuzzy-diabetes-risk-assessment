clear; clc;

%% ---------- Load and clean data ----------
here = fileparts(mfilename('fullpath'));     % .../Project/code
root = fileparts(here);                      % .../Project
data = readtable(fullfile(root, 'data', 'pima-indians-diabetes.csv'), 'ReadVariableNames', false);
data.Properties.VariableNames = {'Pregnancies','Glucose','BloodPressure', ...
    'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'};

for v = {'Glucose','BloodPressure','BMI'}
    col = data.(v{1});
    col(col==0) = median(col(col~=0));
    data.(v{1}) = col;
end

X = data{:, {'Glucose','BMI','Age','BloodPressure'}};
y = data.Outcome;

%% ---------- Train / test split (70/30, stratified) ----------
rng(1);
cv  = cvpartition(y, 'HoldOut', 0.3);
Xtr = X(training(cv),:);  ytr = y(training(cv));
Xte = X(test(cv),:);      yte = y(test(cv));
fprintf('Train: %d rows, Test: %d rows\n', numel(ytr), numel(yte));

warning('off','all');      % hide "No rules fired" warnings

%% ---------- BEFORE: hand-designed FIS ----------
thr  = 30:1:95;
fis0 = buildFIS([]);                              % original membership functions
s0   = evalfis(fis0, Xtr);
[thr0, trAcc0] = bestThreshold(s0, ytr, thr);     % threshold chosen on TRAIN only
teAcc0 = mean((evalfis(fis0, Xte) > thr0) == yte) * 100;
fprintf('BEFORE GA -> train: %.2f%%, test: %.2f%% (threshold %d)\n', trAcc0, teAcc0, thr0);

%% ---------- GA: tune Glucose and BMI membership functions + threshold ----------
% p = [gA gD1 gD2 gD3  bA bD1 bD2 bD3  threshold]
lb = [ 80  5  5  5   22  2  2  2   30];
ub = [110 25 25 25   30  8  8  8   90];

fitness = @(p) -fitnessAcc(p, Xtr, ytr);
opts = optimoptions('ga', 'PopulationSize', 30, 'MaxGenerations', 40, ...
                    'Display', 'iter');
[pbest, fval] = ga(fitness, 9, [], [], [], [], lb, ub, [], opts);

%% ---------- AFTER: GA-tuned FIS ----------
fis1   = buildFIS(pbest(1:8));
trAcc1 = mean((evalfis(fis1, Xtr) > pbest(9)) == ytr) * 100;
teAcc1 = mean((evalfis(fis1, Xte) > pbest(9)) == yte) * 100;
fprintf('\nAFTER GA  -> train: %.2f%%, test: %.2f%% (threshold %.1f)\n', trAcc1, teAcc1, pbest(9));
fprintf('Improvement on test: %+.2f points\n', teAcc1 - teAcc0);

disp('Best parameters:'); disp(pbest);

%% ---------- Plots for the report ----------
figure; plotmf(fis0,'input',1); title('Glucose MFs - before GA');
figure; plotmf(fis1,'input',1); title('Glucose MFs - after GA');
figure; plotmf(fis0,'input',2); title('BMI MFs - before GA');
figure; plotmf(fis1,'input',2); title('BMI MFs - after GA');
figure; bar([teAcc0 teAcc1]); set(gca,'XTickLabel',{'Before GA','After GA'});
ylabel('Test accuracy (%)'); title('Test accuracy comparison'); grid on;

warning('on','all');

%% ================= Local functions =================
function acc = fitnessAcc(p, X, y)
    fis = buildFIS(p(1:8));
    acc = mean((evalfis(fis, X) > p(9)) == y) * 100;
end

function [bestT, bestA] = bestThreshold(score, y, thr)
    a = zeros(size(thr));
    for i = 1:numel(thr)
        a(i) = mean((score > thr(i)) == y) * 100;
    end
    [bestA, k] = max(a);
    bestT = thr(k);
end

function fis = buildFIS(p)
    fis = mamfis('Name','DiabetesRisk');

    % ---- Glucose ----
    fis = addInput(fis,[40 200],'Name','Glucose');
    if isempty(p)
        fis = addMF(fis,'Glucose','trapmf',[40 40 90 110],'Name','Low');
        fis = addMF(fis,'Glucose','trimf',[100 125 150],'Name','Medium');
        fis = addMF(fis,'Glucose','trapmf',[140 165 200 200],'Name','High');
    else
        a = p(1); b = a + p(2); c = b + p(3); d = c + p(4);
        fis = addMF(fis,'Glucose','trapmf',[40 40 a b],'Name','Low');
        fis = addMF(fis,'Glucose','trimf',[a (b+c)/2 d],'Name','Medium');
        fis = addMF(fis,'Glucose','trapmf',[c d 200 200],'Name','High');
    end

    % ---- BMI ----
    fis = addInput(fis,[15 70],'Name','BMI');
    if isempty(p)
        fis = addMF(fis,'BMI','trapmf',[15 15 23 27],'Name','Normal');
        fis = addMF(fis,'BMI','trimf',[25 30 35],'Name','Overweight');
        fis = addMF(fis,'BMI','trapmf',[32 38 70 70],'Name','Obese');
    else
        a = p(5); b = a + p(6); c = b + p(7); d = c + p(8);
        fis = addMF(fis,'BMI','trapmf',[15 15 a b],'Name','Normal');
        fis = addMF(fis,'BMI','trimf',[a (b+c)/2 d],'Name','Overweight');
        fis = addMF(fis,'BMI','trapmf',[c d 70 70],'Name','Obese');
    end

    % ---- Age (fixed) ----
    fis = addInput(fis,[20 85],'Name','Age');
    fis = addMF(fis,'Age','trapmf',[20 20 30 40],'Name','Young');
    fis = addMF(fis,'Age','trimf',[35 48 60],'Name','Middle');
    fis = addMF(fis,'Age','trapmf',[55 65 85 85],'Name','Old');

    % ---- Blood pressure (fixed) ----
    fis = addInput(fis,[20 125],'Name','BloodPressure');
    fis = addMF(fis,'BloodPressure','trapmf',[20 20 65 80],'Name','Low');
    fis = addMF(fis,'BloodPressure','trimf',[70 80 90],'Name','Normal');
    fis = addMF(fis,'BloodPressure','trapmf',[85 100 125 125],'Name','High');

    % ---- Output ----
    fis = addOutput(fis,[0 100],'Name','Risk');
    fis = addMF(fis,'Risk','trimf',[0 0 40],'Name','Low');
    fis = addMF(fis,'Risk','trimf',[25 45 65],'Name','Medium');
    fis = addMF(fis,'Risk','trimf',[60 100 100],'Name','High');

    % ---- Rules (19) ----
    rules = [
     "Glucose==High => Risk=High"
     "Glucose==High & BMI==Obese => Risk=High"
     "Glucose==High & BMI==Overweight => Risk=High"
     "Glucose==High & Age==Old => Risk=High"
     "Glucose==High & Age==Middle => Risk=High"
     "Glucose==High & BloodPressure==High => Risk=High"
     "Glucose==Medium & BMI==Obese => Risk=High"
     "Glucose==Medium & BMI==Overweight => Risk=Medium"
     "Glucose==Medium & BMI==Normal => Risk=Low"
     "Glucose==Medium & Age==Old => Risk=Medium"
     "Glucose==Medium & Age==Middle => Risk=Medium"
     "Glucose==Medium & Age==Young => Risk=Low"
     "Glucose==Medium & BloodPressure==High => Risk=Medium"
     "Glucose==Low & BMI==Normal => Risk=Low"
     "Glucose==Low & BMI==Overweight => Risk=Low"
     "Glucose==Low & BMI==Obese => Risk=Medium"
     "Glucose==Low & Age==Young => Risk=Low"
     "Glucose==Low & Age==Old => Risk=Medium"
     "Glucose==Low & BloodPressure==Normal => Risk=Low"
    ];
    fis = addRule(fis, rules);
end
