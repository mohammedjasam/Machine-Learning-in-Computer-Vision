function[acc, min_D, I, comp] = comAccu(dis,nTest,labels1,labels2)
    [min_D, I] = min(dis,[],1);

    %nTest = size(Ims,2);
    count = 0;
    count1 = 0;
    comp=zeros(nTest,2);
    for i = 1:nTest
        comp(i,:) = [labels2(i,1), labels1(I(i),1)]; 
        if(labels2(i,1) == labels1(I(i),1))
            disp(i);
            disp(labels2(i));
            disp(labels1(I(i)));
            disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++');
            count = count +1;
        end
        
        if(labels2(i) == labels1(I(i)))
            disp(i);
            disp(labels2(i));
            disp(labels1(I(i)));
            disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++');
            count1 = count1 +1;
    end
    
    %Accuracy
    acc = count/nTest;
    disp(count);
    disp('==================================');
    disp(count1);
%     disp(nTest);
end