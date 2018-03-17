function [P_O, P_S, P_B, R_O, R_S, R_B, F1_O, F1_S, F1_B] = qual_measure (op, gt)
    
    nrow = min([size(op, 1), size(gt, 1)]);
    ncol = min([size(op, 2), size(gt, 2)]);
    
    match_O = 0;
    match_S = 0;
    match_B = 0;
    nonmatch = 0;
    for i = 1:nrow
        for j = 1:ncol
            if op(i,j) == gt(i,j)
                if gt(i,j) == 0
                    match_O = match_O + 1;
                elseif gt(i,j) == 1
                    match_S = match_S + 1;
                else
                    match_B = match_B + 1;
                end
            else
                nonmatch = nonmatch + 1;
            end
                
        end
    end
    P_O = match_O/size(find(op==0), 1)
    P_S = match_S/size(find(op==1), 1)
    P_B = match_B/size(find(op==2), 1)
    R_O = match_O/size(find(gt==0), 1)
    R_S = match_S/size(find(gt==1), 1)
    R_B = match_B/size(find(gt==2), 1)
    F1_O = (2*P_O*R_O)/(P_O+R_O)
    F1_S = (2*P_S*R_S)/(P_S+R_S)
    F1_B = (2*P_B*R_B)/(P_B+R_B)
end
