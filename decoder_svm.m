function [OUT] = decoder_svm(eStruct,PARAM)

B = eStruct.Behavior; B1 = PARAM.Sample_1; B2 = PARAM.Sample_2;
W = PARAM.ExclusionWindow;
fps = PARAM.Fps;
Len = PARAM.Length;
DM = zscore(eStruct.Ms.FiltTraces(:, :));

jj_1 = find(strcmp(B.EventNames, B1));
jj_2 = find(strcmp(B.EventNames, B2));
ON_1 = B.Onset{jj_1}; OFF_1 = B.Offset{jj_1};
ON_2 = B.Onset{jj_2}; OFF_2 = B.Offset{jj_2}; 

ON_all = [ON_1, ON_2];
OFF_all = [OFF_1, OFF_2];
Y = [zeros(1,length(ON_1)) ones(1,length(ON_2))]; OUT.Label = Y;
OUT.Sample_1 = sum(Y==0); OUT.Sample_2 = sum(Y==1);
if sum(Y == 0) == 0 || sum(Y == 1) == 0 
    fprintf('Warning: %s, %d bout; %s %d bout\n', B1, length(ON_1), B2, length(ON_2));
    OUT.Scores = NaN; OUT.PredLabel = NaN; OUT.auROC = NaN; 
    return
end

%% Shuffle control 
if PARAM.Shuffle == 1
    shift = fps*300 + randperm(size(DM, 1)-2*fps*300,1); % shift at least 5 min in both direction
    for u = 1:size(DM,2); DM(:,u) = circshift(DM(:,u), shift); end
end

%% Leave-one-out procedure
for xx = 1:length(ON_all) 
    test_jj = zeros(1,size(DM,1));
    test_jj(max(ON_all(xx)-W, 1):min(OFF_all(xx)+W, size(DM,1))) = 1;
    
    X_train = cell(1,2); 
    
    % Sample 1 for training   
    jj = find(strcmp(B.EventNames, B1));                                       
    BV = B.LogicalVecs{jj};
    ON = B.Onset{jj}; OFF = B.Offset{jj};
    
    ind = zeros(1, length(ON));
    for i = 1:length(ON)
        if sum(test_jj(ON(i):OFF(i))) == 0 
            ind(i) = 1;
        end
    end
    ON_1 = ON(ind == 1); OFF_1 = OFF(ind == 1);
 
    % Sample 2 for training
    jj = find(strcmp(B.EventNames, B1));                                       
    BV = B.LogicalVecs{jj};
    ON = B.Onset{jj}; OFF = B.Offset{jj};
    
    ind = zeros(1, length(ON));
    for i = 1:length(ON)
        if sum(test_jj(ON(i):OFF(i))) == 0 
            ind(i) = 1;
        end
    end
    ON_2 = ON(ind == 1); OFF_2 = OFF(ind == 1);
     
    for tt = 1:length(ON_1)
        X_train{1}(tt,:) = mean(DM(ON_1(tt):min(OFF_1(tt), ON_1(tt)+Len*fps),:),1); 
    end
    
    for tt = 1:length(ON_2)
        X_train{2}(tt,:) = mean(DM(ON_2(tt):min(OFF_2(tt), ON_2(tt)+Len*fps),:),1); 
    end
            
    if size(X_train{1}, 1) <= 3 || size(X_train{2}, 1) <= 3
        fprintf('Warning: %dth fold, low training size: %s, %d; %s, %d\n', xx, B1, size(X_train{1}, 1), B2, size(X_train{2}, 1))
        Y(xx) = NaN; Yh(xx) = NaN; scores{xx} = [NaN, NaN];
        continue
    end
    
    if ~isempty(X_train{1}) && ~isempty(X_train{2})             
        if PARAM.PLS == 1
            [L, ~, ~, ~, ~, V] = plsregress(vertcat(X_train{:,:}),[zeros(size(X_train{1},1),1); ones(size(X_train{2},1),1)]); 
            cumvar = cumsum(V(1, :))/sum(V(1, :)); num_pls = find(cumvar >= PARAM.PLSvar, 1); 
            for n = 1:2; X_train{n} = X_train{n}*L(:,1:num_pls); end
        end
                
        Y_train = [zeros(size(X_train{1},1),1); ones(size(X_train{2},1),1)];
        X_train = vertcat(X_train{:,:}); 
        
        X_test = mean(DM(ON_all(xx):min(OFF_all(xx), ON_all(xx)+Len*fps),:));
        if PARAM.PLS == 1
            X_test = X_test*L(:, 1:num_pls);
        end
        
        try
            mdl = fitcsvm(X_train,Y_train, 'KernelFunction', PARAM.Kernel, 'KernelScale', PARAM.KernelScale); 
            [Yh(xx), scores{xx}] = predict(mdl, X_test);
        catch
            Y(xx) = NaN;
            Yh(xx) = NaN;
        end
    else
        Y(xx) = NaN;
        Yh(xx) = NaN;
    end    
end

S = vertcat(scores{:,:}); 
OUT.Scores = S(:,2);
OUT.PredLabel = Yh;

G = Y(~isnan(Y));
S = S(~isnan(Y), :);
P = Yh(~isnan(Y));

if ~isempty(G) && ~isempty(P) && length(unique(G)) == 2
    [~,~,~,OUT.auROC] = perfcurve(G,S(:,2),1);
else
    OUT.auROC = NaN;
end
end
