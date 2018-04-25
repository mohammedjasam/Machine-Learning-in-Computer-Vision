function viz(Inference1, Inference2, Inference3, Inference4, Inference5, Inference6, Name1, Name2, Name3, Name4, Name5, Name6, WTest, NumTestFace, NumTestBackground)
         
    XAxis = [];
    for i = 1 : (size(WTest, 1))
        XAxis = [XAxis; i];
    end
    
    figure();
    % Visualization
    subplot(2,3,1);   
    plot(XAxis, WTest); hold on;
    plot(XAxis, Inference1); hold off;
    title(sprintf('%s\n%s', Name1, 'Ground Truth vs. Prediction for Faces and Background'));
    xlabel('Images');
    ylabel('Prediction Probability Score');
    legend('Ground Truth', 'Prediction');
    
    subplot(2,3,2);   
    plot(XAxis, WTest); hold on;
    plot(XAxis, Inference2); hold off;
    title(sprintf('%s\n%s', Name2, 'Ground Truth vs. Prediction for Faces and Background'));
    xlabel('Images');
    ylabel('Prediction Probability Score');
    legend('Ground Truth', 'Prediction');
    
    subplot(2,3,3);   
    plot(XAxis, WTest); hold on;
    plot(XAxis, Inference3); hold off;
    title(sprintf('%s\n%s', Name3, 'Ground Truth vs. Prediction for Faces and Background'));
    xlabel('Images');
    ylabel('Prediction Probability Score');
    legend('Ground Truth', 'Prediction');
    
    subplot(2,3,4);   
    plot(XAxis, WTest); hold on;
    plot(XAxis, Inference4); hold off;
    title(sprintf('%s\n%s', Name4, 'Ground Truth vs. Prediction for Faces and Background'));
    xlabel('Images');
    ylabel('Prediction Probability Score');
    legend('Ground Truth', 'Prediction');
    
    subplot(2,3,5);   
    plot(XAxis, WTest); hold on;
    plot(XAxis, Inference5); hold off;
    title(sprintf('%s\n%s', Name5, 'Ground Truth vs. Prediction for Faces and Background'));
    xlabel('Images');
    ylabel('Prediction Probability Score');
    legend('Ground Truth', 'Prediction');
    
    subplot(2,3,6);   
    plot(XAxis, WTest); hold on;
    plot(XAxis, Inference6); hold off;
    title(sprintf('%s\n%s', Name6, 'Ground Truth vs. Prediction for Faces and Background'));
    xlabel('Images');
    ylabel('Prediction Probability Score');
    legend('Ground Truth', 'Prediction');

    
    
end