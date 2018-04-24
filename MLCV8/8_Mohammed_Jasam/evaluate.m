function [MissDetection, FalseAlarm] = evaluate(WTest, Inference, NumTestFace, NumTestBackground)

    % Classifying
    Inference(Inference >= 0.5) = 1;
    Inference(Inference < 0.5) = 0;

    % Evaluation for Face first then Background
    MissDetection = sum(abs(WTest(1 : NumTestFace)' - Inference(1 : NumTestFace))) / NumTestFace;
    FalseAlarm = sum(abs(WTest(NumTestFace + 1 : NumTestFace + NumTestBackground)' - Inference(NumTestFace + 1 : NumTestFace + NumTestBackground))) / NumTestBackground;
    
%     % Evaluation for Background first then Face
%     MissDetection = sum(abs(WTest(NumTestBackground + 1 : NumTestBackground + NumTestFace)' - Inference(NumTestBackground + 1 : NumTestBackground + NumTestFace))) / NumTestFace;
%     FalseAlarm = sum(abs(WTest(1 : NumTestBackground)' - Inference(1 : NumTestBackground))) / NumTestBackground;

end 
