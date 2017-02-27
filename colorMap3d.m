function colorMap3d(Frame,a1,a2)
%myGrid=[linspace(-1,1,20);linspace(-1,1,20);linspace(84.4,86.4,20)]
x=zeros(20,20,20);
y=zeros(20,20,20);
z=zeros(20,20,20);
map3d=zeros(20,20,20);
for i=1:20
    for j=1:20
        for k=1:20
            point=[0,0,0,(i-10)*0.1,(j-10)*0.1,84.4+(k-10)*0.1];
            x(i,j,k)=(i-10)*0.1;
            y(i,j,k)=(j-10)*0.1;
            z(i,j,k)=84.4+(k-10)*0.1;
            map3d(i,j,k)=BrightnesScalarization(Frame,a1,a2,point);
        end
        tmp=((i-1)/20)*100+((j-1)/20)*10
    end
end
%map3d
cVals = unique(map3d);
for i = 1:numel(cVals)                     % For every one of those unique values
    indices = find(map3d == cVals(i));         % Find the corresponding indices
    scatter3(x(indices),y(indices),z(indices),100,'filled') % Plot
    hold on
end