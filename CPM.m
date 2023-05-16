function prediction_result = CPM(behaviour_result,fc_matrix)

nBehav = size(behaviour_result,2);
[nchan,~,nfreq,nSubject] = size(fc_matrix);
fs_thre = 0.05; % threshold for feature selection 

behav_pred_pos = zeros(nSubject,nBehav);
behav_pred_neg = zeros(nSubject,nBehav);

for leftout = 1:nSubject
    fprintf('Leaving out subject # %1.0f \n',leftout);
    
    train_mats = fc_matrix;
    test_mats = fc_matrix(:,:,:,leftout);
    train_mats(:,:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,4));
    
    train_behav = behaviour_result;
    train_behav(leftout,:) = [];
    
    % correlate all edges with behaviour result     
    [r_mat,p_mat] = corr(train_vcts',train_behav);
    r_mat = reshape(r_mat,nchan,nchan,nfreq,[]);
    p_mat = reshape(p_mat,nchan,nchan,nfreq,[]);
    
    pos_mask = zeros(nchan,nchan,nfreq,nBehav);
    neg_mask = zeros(nchan,nchan,nfreq,nBehav);
    
    pos_edges = find(r_mat>0 & p_mat<fs_thre);
    neg_edges = find(r_mat<0 & p_mat<fs_thre);
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1;
    % sigmoidal weighting 
%     pos_edges = find(r_mat>0);
%     neg_edges = find(r_mat<0);
%     T = tinv(fs_thre/2,nSubject-1-2);
%     R = sqrt(T^2/(nSubject-1-2+T^2));
%     pos_mask(pos_edges) = sigmf(r_mat(pos_edges),[3/R,R/3]);
%     neg_mask(neg_edges) = sigmf(r_mat(neg_edges),[-3/R,R/3]);
    
    % get sum of all edges in train subjects
    train_sumpos = zeros(nSubject-1,nBehav);
    train_sumneg = zeros(nSubject-1,nBehav);
    
    for bidx = 1:nBehav
        for ss = 1:size(train_sumpos,1)
            % sum over the four frequency bands 
            train_sumpos(ss,bidx) = sum(sum(sum(train_mats(:,:,:,ss).*pos_mask(:,:,:,bidx))))/2;
            train_sumneg(ss,bidx) = sum(sum(sum(train_mats(:,:,:,ss).*neg_mask(:,:,:,bidx))))/2;
        end
        
        fit_pos(:,bidx) = polyfit(train_sumpos(:,bidx),train_behav(:,bidx),1);
        fit_neg(:,bidx) = polyfit(train_sumneg(:,bidx),train_behav(:,bidx),1);
        
        test_sumpos(bidx) = sum(sum(sum(test_mats.*pos_mask(:,:,:,bidx))));
        test_sumneg(bidx) = sum(sum(sum(test_mats.*neg_mask(:,:,:,bidx))));
        
        behav_pred_pos(leftout,bidx) = fit_pos(1,bidx)*test_sumpos(bidx)+fit_pos(2,bidx);
        behav_pred_neg(leftout,bidx) = fit_neg(1,bidx)*test_sumneg(bidx)+fit_neg(2,bidx);
    end
end

for bidx = 1:nBehav
    [R_pos(bidx),P_pos(bidx)] = corr(behav_pred_pos(:,bidx),behaviour_result(:,bidx));
    [R_neg(bidx),P_neg(bidx)] = corr(behav_pred_neg(:,bidx),behaviour_result(:,bidx));
end

prediction_result.R_pos = R_pos;
prediction_result.R_neg = R_neg;
prediction_result.P_pos = P_pos;
prediction_result.P_neg = P_neg;
