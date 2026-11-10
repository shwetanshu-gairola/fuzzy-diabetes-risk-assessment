clear; clc;

%% ---------- Load and clean data ----------
here   = fileparts(mfilename('fullpath'));   % .../Project/code
root   = fileparts(here);                    % .../Project
resDir = fullfile(root, 'results');
figDir = fullfile(root, 'figures');
if ~exist(resDir,'dir'), mkdir(resDir); end
if ~exist(figDir,'dir'), mkdir(figDir); end
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

warning('off','all');

%% ---------- Settings ----------
K   = 5;                          % outer folds: every patient is tested exactly once
thr = 30:1:95;
nR  = 19;                         % number of rules

% MF parameters (16): [Glucose a d1 d2 d3 | BMI a d1 d2 d3 | Age a d1 d2 d3 | BP a d1 d2 d3]
lbMF = [ 80  5  5  5   22 2 2 2   28  4  4  4   60 3 3 3];
ubMF = [110 25 25 25   30 8 8 8   40 15 15 15   80 12 12 12];
% rule weights (19)
lbW = 0.2*ones(1,nR);  ubW = ones(1,nR);

opts = optimoptions('ga', 'PopulationSize', 30, 'MaxGenerations', 25, 'Display', 'off');

%% ---------- Outer 5-fold cross-validation ----------
rng(1);
cv = cvpartition(y, 'KFold', K);

predBase = zeros(size(y));  scoreBase = zeros(size(y));
predA    = zeros(size(y));  scoreA    = zeros(size(y));
predB    = zeros(size(y));  scoreB    = zeros(size(y));
foldAcc  = zeros(K, 3);     foldAUC   = zeros(K, 3);

for k = 1:K
    tr = training(cv, k);  te = test(cv, k);
    Xtr = X(tr,:); ytr = y(tr);  Xte = X(te,:); yte = y(te);

    % ---- Baseline: hand-designed FIS ----
    f0 = buildFIS([], false);
    t0 = bestThreshold(evalfis(f0, Xtr), ytr, thr);
    s0 = evalfis(f0, Xte);
    predBase(te) = s0 > t0;  scoreBase(te) = s0;

    % ---- Variant A: GA tunes membership functions (fitness = train AUC) ----
    rng(100+k);
    pA = ga(@(p) -aucScore(evalfis(buildFIS(p,false), Xtr), ytr), 16, [],[],[],[], lbMF, ubMF, [], opts);
    fA = buildFIS(pA, false);
    tA = bestThreshold(evalfis(fA, Xtr), ytr, thr);
    sA = evalfis(fA, Xte);
    predA(te) = sA > tA;  scoreA(te) = sA;

    % ---- Variant B: GA tunes membership functions + rule weights ----
    rng(200+k);
    pB = ga(@(p) -aucScore(evalfis(buildFIS(p,true), Xtr), ytr), 16+nR, [],[],[],[], [lbMF lbW], [ubMF ubW], [], opts);
    fB = buildFIS(pB, true);
    tB = bestThreshold(evalfis(fB, Xtr), ytr, thr);
    sB = evalfis(fB, Xte);
    predB(te) = sB > tB;  scoreB(te) = sB;

    foldAcc(k,:) = [mean(predBase(te)==yte) mean(predA(te)==yte) mean(predB(te)==yte)] * 100;
    foldAUC(k,:) = [aucScore(s0,yte) aucScore(sA,yte) aucScore(sB,yte)];
    fprintf('Fold %d: acc  base %.2f | A %.2f | B %.2f\n', k, foldAcc(k,:));
end

%% ---------- Summary ----------
names = {'Hand-designed','GA: MFs only (A)','GA: MFs + rule weights (B)'};
pooledAcc = [mean(predBase==y) mean(predA==y) mean(predB==y)] * 100;
pooledAUC = [aucScore(scoreBase,y) aucScore(scoreA,y) aucScore(scoreB,y)];

fprintf('\n===== %d-fold cross-validation, every patient tested once =====\n', K);
for i = 1:3
    fprintf('%-28s accuracy %.2f%% (fold mean %.2f +/- %.2f) | AUC %.3f\n', ...
        names{i}, pooledAcc(i), mean(foldAcc(:,i)), std(foldAcc(:,i)), pooledAUC(i));
end

T = array2table([pooledAcc; mean(foldAcc); std(foldAcc); pooledAUC]', ...
    'VariableNames', {'PooledAcc','FoldMeanAcc','FoldStdAcc','AUC'}, 'RowNames', names);
disp(T);
writetable(T, fullfile(resDir, 'ga_v2_results.csv'), 'WriteRowNames', true);

figure;
bar(foldAcc); grid on;
xlabel('Fold'); ylabel('Test accuracy (%)'); ylim([60 90]);
legend(names, 'Location','southoutside', 'Orientation','horizontal');
title('Test accuracy per fold');
exportgraphics(gcf, fullfile(figDir, 'ga_v2_folds.png'), 'Resolution', 200);

warning('on','all');

%% ================= Local functions =================
function a = aucScore(s, y)
    r  = tiedrank(s);
    n1 = sum(y==1);  n0 = sum(y==0);
    a  = (sum(r(y==1)) - n1*(n1+1)/2) / (n1*n0);
end

function bestT = bestThreshold(score, y, thr)
    a = zeros(size(thr));
    for i = 1:numel(thr)
        a(i) = mean((score > thr(i)) == y);
    end
    [~, k] = max(a);
    bestT = thr(k);
end

function fis = buildFIS(p, useW)
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

    % ---- Age ----
    fis = addInput(fis,[20 85],'Name','Age');
    if isempty(p)
        fis = addMF(fis,'Age','trapmf',[20 20 30 40],'Name','Young');
        fis = addMF(fis,'Age','trimf',[35 48 60],'Name','Middle');
        fis = addMF(fis,'Age','trapmf',[55 65 85 85],'Name','Old');
    else
        a = p(9); b = a + p(10); c = b + p(11); d = c + p(12);
        fis = addMF(fis,'Age','trapmf',[20 20 a b],'Name','Young');
        fis = addMF(fis,'Age','trimf',[a (b+c)/2 d],'Name','Middle');
        fis = addMF(fis,'Age','trapmf',[c d 85 85],'Name','Old');
    end

    % ---- Blood pressure ----
    fis = addInput(fis,[20 125],'Name','BloodPressure');
    if isempty(p)
        fis = addMF(fis,'BloodPressure','trapmf',[20 20 65 80],'Name','Low');
        fis = addMF(fis,'BloodPressure','trimf',[70 80 90],'Name','Normal');
        fis = addMF(fis,'BloodPressure','trapmf',[85 100 125 125],'Name','High');
    else
        a = p(13); b = a + p(14); c = b + p(15); d = c + p(16);
        fis = addMF(fis,'BloodPressure','trapmf',[20 20 a b],'Name','Low');
        fis = addMF(fis,'BloodPressure','trimf',[a (b+c)/2 d],'Name','Normal');
        fis = addMF(fis,'BloodPressure','trapmf',[c d 125 125],'Name','High');
    end

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

    if useW
        w = p(17:35);
        for i = 1:numel(fis.Rules)
            fis.Rules(i).Weight = w(i);
        end
    end
end
