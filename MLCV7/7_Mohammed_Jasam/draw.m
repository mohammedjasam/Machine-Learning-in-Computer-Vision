function draw(GT, Inference, Title)
    figure();
    plot(GT);
    hold on
    plot(Inference);
    legend('Ground Truth', Title);
    title(Title);
    xlabel('Ground Truth Angle');
    ylabel('Inference Angle');
    hold off
end