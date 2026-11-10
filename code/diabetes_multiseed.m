clear; clc;

%% ---------- Load and clean data ----------
here   = fileparts(mfilename('fullpath'));   % .../Project/code
root   = fileparts(here);                    % .../Project
resDir = fullfile(root, 'results');
if ~exist(resDir,'dir'), mkdir(resDir); end
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

%% ---------- Repeat the experiment over several random splits ----------
seeds = 1:5;
thr   = 30:1:95;
lb = [ 80  5  5  5   22  2  2  2   30];
ub = [110 25 25 25   30  8  8  8   90];

res = zeros(numel(seeds), 3);      % [before-test, after-test, improvement]
allP = zeros(numel(seeds), 9);

for k = 1:numel(seeds)
    rng(seeds(k));
    cv  = cvpartition(y, 'HoldOut', 0.3);
    Xtr = X(training(cv),:);  ytr = y(training(cv));
    Xte = X(test(cv),:);      yte = y(test(cv));

    % BEFORE: hand-designed FIS, threshold chosen on train
    fis0 = buildFIS([]);
    [thr0, ~] = bestThreshold(evalfis(fis0, Xtr), ytr, thr);
    te0 = mean((evalfis(fis0, Xte) > thr0) == yte) * 100;

    % AFTER: GA-tuned FIS
    opts = optimoptions('ga', 'PopulationSize', 30, 'MaxGenerations', 40, 'Display', 'off');
    pbest = ga(@(p) -fitnessAcc(p, Xtr, ytr), 9, [], [], [], [], lb, ub, [], opts);
    fis1 = buildFIS(pbest(1:8));
    te1  = mean((evalfis(fis1, Xte) > pbest(9)) == yte) * 100;

    res(k,:)  = [te0 te1 te1-te0];
    allP(k,:) = pbest;
    fprintf('Seed %d: before %.2f%%, after %.2f%%, change %+.2f\n', seeds(k), te0, te1, te1-te0);
end

%% ---------- Summary ----------
fprintf('\n===== Summary over %d splits (test accuracy) =====\n', numel(seeds));
fprintf('Before GA: %.2f%% +/- %.2f\n', mean(res(:,1)), std(res(:,1)));
fprintf('After GA : %.2f%% +/- %.2f\n', mean(res(:,2)), std(res(:,2)));
fprintf('Mean improvement: %+.2f points (GA better in %d of %d splits)\n', ...
    mean(res(:,3)), sum(res(:,3) > 0), numel(seeds));

T = array2table(res, 'VariableNames', {'BeforeGA','AfterGA','Change'});
T.Seed = seeds(:);
T = T(:, {'Seed','BeforeGA','AfterGA','Change'});
disp(T);
writetable(T, fullfile(resDir, 'multiseed_results.csv'));

figure;
bar(seeds, res(:,1:2)); grid on;
xlabel('Split (seed)'); ylabel('Test accuracy (%)');
legend('Before GA','After GA','Location','southoutside','Orientation','horizontal');
title('Test accuracy across random splits');
ylim([60 85]);

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
