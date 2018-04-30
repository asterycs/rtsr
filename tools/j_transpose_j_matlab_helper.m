% 3d tests

a = sym('a', [1 8], 'real');
b = sym('b', [1 8], 'real');
g = sym('g', [1 8], 'real');
z = sym('z', [1 8], 'real');
Js = [a(1) b(1) 0 g(1) 0 0 0 0 0; 0 g(2) 0 b(2) a(2) 0 0 0 0; 0 a(3) b(3) 0 g(3) 0 0 0 0; 0 0 g(4) 0 b(4) a(4) 0 0 0;0 0 0 a(5) b(5) 0 g(5) 0 0; 0 0 0 0 g(6) 0 b(6) a(6) 0;0 0 0 0 a(7) b(7) 0 g(7) 0; 0 0 0 0 0 g(8) 0 b(8) a(8)];
Js_even = Js;
Js_even(2,:) = 0;
Js_even(4,:) = 0;
Js_even(6,:) = 0;
Js_even(8,:) = 0;

Js_uneven = Js;
Js_uneven(1,:) = 0;
Js_uneven(3,:) = 0;
Js_uneven(5,:) = 0;
Js_uneven(7,:) = 0;

Jzs = Js'*z';

a = ones(1,8)*0.33;
b = ones(1,8)*0.33;
g = ones(1,8)*0.34;

Jn = [a(1) b(1) 0 g(1) 0 0 0 0 0; 0 g(2) 0 a(2) b(2) 0 0 0 0; 0 a(3) b(3) 0 g(3) 0 0 0 0; 0 0 g(4) 0 b(4) a(4) 0 0 0;0 0 0 a(5) b(5) 0 g(5) 0 0; 0 0 0 0 g(6) 0 a(6) b(6) 0;0 0 0 0 a(7) b(7) 0 g(7) 0; 0 0 0 0 0 g(8) 0 b(8) a(8)];

Jn_even = Jn;

for i=2:2:size(Jn_even,1)
    Jn_even(i,:) = 0;
end

Jn_uneven = Jn;

for i=1:2:size(Jn_uneven,1)
    Jn_uneven(i,:) = 0;
end
