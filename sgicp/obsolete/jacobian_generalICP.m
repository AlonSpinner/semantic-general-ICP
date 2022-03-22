t = sym('t',[2,1],'real');
theta = sym('theta','real');
R = [cos(theta), - sin(theta); sin(theta), cos(theta)];
%R = sym('R',[2,2],'real');
a = sym('a',[2,1],'real');
b = sym('b',[2,1],'real');
Sa = sym('Sa',[2,2],'real');
Sb = sym('Sb',[2,2],'real');

f = sym('f',[2,2],'real');

d = b - R*a - t;
M = Sb + R * Sb * R';
e = d' * inv(M) *d;