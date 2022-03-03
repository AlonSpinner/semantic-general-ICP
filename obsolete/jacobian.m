x = sym('x',[3,1]); % x,y,theta   %pose is stored as 
l = sym('l',[2,1]); % x,y
z = sym('z',[2,1]); % x,y

R = [cos(x(3)), sin(x(3));
      -sin(x(3)), cos(x(3))]; %C2W
t = [x(1); %W_W2C
    x(2)];
h = R*l+t; %h(x,l)

err = h - z;
Jl = jacobian(h,l);
Jx = jacobian(h,x);


 