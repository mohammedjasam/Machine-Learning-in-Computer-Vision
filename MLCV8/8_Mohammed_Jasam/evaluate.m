function [MissDetection, FalseAlarm] = evaluate(WTest, Inference, NumTestFace, NumTestBackground)

    % Classifying
    Inference(Inference >= 0.5) = 1;
    Inference(Inference < 0.5) = 0;

    % Evaluation
    MissDetection = sum(abs(WTest(1 : NumTestFace)' - Inference(1 : NumTestFace))) / NumTestFace;
    FalseAlarm = sum(abs(WTest(NumTestFace + 1 : NumTestFace + NumTestBackground)' - Inference(NumTestFace + 1 : NumTestFace + NumTestBackground))) / NumTestBackground;

end
