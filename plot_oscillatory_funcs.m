% Script for making the plot in Figure 6 of our paper

figure

subplot(1,3,1)
x = linspace(-1,1,4^10);
y = (x+1).*sin(100*(x+1).^2);
plot(x,y)
title('Linear Growth')

subplot(1,3,2)
x = linspace(.01,100,4^10);
y = x.^(-1/4).*sin(2/3 * x.^(3/2));
plot(x,y)
title('Airy')

subplot(1,3,3)
x = linspace(0.01,1,4^10);
y = sin(4./x).*cos(x.^2);
plot(x,y)
title('Chirp')
