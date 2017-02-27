function [Z V]=MBJ(n,x)
a=tic;
for i=1:100
m=0:1:20;
licznik=zeros(1,20,'uint64');
licznik=(x/2).^(2.*m+n).*(-1).^m./(factorial(m).*gamma(m+n+1));
Z=sum(licznik);
end
czas=toc(a)
b=tic;
for i=1:100
	V=besselj(n,x);
end
czas2=toc(b)