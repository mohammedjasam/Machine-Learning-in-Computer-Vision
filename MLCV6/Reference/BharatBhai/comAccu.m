function[acc, min_D, I, comp] = comAccu(dis,nTest,labels1,labels2)
    [min_D, I] = min(dis,[],1);

    %nTest = size(Ims,2);
    count = 0;
    comp=zeros(nTest,2);
    for i = 1:nTest
        comp(i,:) = [labels2(i,1), labels1(I(i),1)]; 
        if(labels2(i,1) == labels1(I(i),1))
            count = count +1;
        end
    end

    %Accuracy
    acc = count/nTest;
end