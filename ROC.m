function R = ROC(F, bTypes, param)

nSims = param.NumberOfSimulations;
B = F.Behavior;
ms = F.Ms;

nUnits = length(ms.goodCellID);
for b = 1:length(bTypes)
    tic
    Z = zscore(ms.FiltTraces(:, ms.goodCellID));
    j = find(strcmp(B.EventNames, bTypes{b}), 1);
    fprintf(1, '\nBehavior %d - %s\n', j, bTypes{b})
    
    Behav = B.LogicalVecs{j};
    B_other = B.LogicalVecs{strcmp(B.EventNames, 'other')}; 
    ind = find((Behav == 1 | B_other == 1));
    Behav = Behav(ind); L = size(Z, 1);

    if ~isempty(j) && nnz(Behav) > 0
        rng(3)
        for ii = 1:nSims
            shft(ii) = randperm(L-120*param.fps,1) + 60*param.fps;
        end
        
        Z_b = Z(ind, :);
        parfor u = 1:nUnits
            [~,~,~,R_obs(u,b)] = perfcurve(Behav, Z_b(:,u), 1);
            for ii = 1:nSims
                Z_sh = circshift(Z(:,u), shft(ii)); Z_sh_b = Z_sh(ind);
                [~,~,~,R_rand{u,b}(ii)] = perfcurve(Behav, Z_sh_b, 1);
            end
        end
    else
        R_obs(:,b) = NaN(nUnits, 1);
        for u = 1:nUnits
            R_rand{u,b} = NaN;
        end
    end
    toc
end

R.obs = R_obs;
R.rand = R_rand;

end