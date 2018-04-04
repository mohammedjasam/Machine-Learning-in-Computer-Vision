function draw(GT, Inference, Title)
    figure();
    plot(GT);
    hold on
    plot(Inference);
    legend('Ground Truth', Title);
    title(Title);
    xlabel('Image in Testing Dataset');
    ylabel('Angle of Rotation');
    hold off
end