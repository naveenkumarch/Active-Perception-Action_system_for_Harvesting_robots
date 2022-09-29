load Joint_angles_data.txt;
load Joint_angles_data_2.txt;
data = Joint_angles_data;
data_2 = Joint_angles_data_2; 

%x = data(1,:);
%y = data(2,:);
%z = data(3,:);
%xyz_values = data(:,1:3);
xyz_values = [data(:,1:3);data_2(:,1:3)];
%Joints_data = data(:,4:9);
Joints_data = [data(:,4:9);data_2(:,4:9)];

[R,Q] = size(xyz_values);

xyz_cam = zeros(R,Q);

for row = 1:R
   xyz_cam(row,1) = ((xyz_values(row,3)*1000)+83);
   xyz_cam(row,2) = ((xyz_values(row,2)*1000)+84);
   xyz_cam(row,3) = ((-xyz_values(row,1)*1000)+9);
end
xyz_values = xyz_values*1000;
save Inputs8.txt xyz_cam -ascii 
save Outputs8.txt Joints_data -ascii
save Inputs7.txt xyz_values -ascii

writematrix(xyz_cam,'Inputs7_de.txt')
type 'Inputs7_de.txt'
writematrix(xyz_values,'Outputs7_de.txt')
type 'Outputs7_de.txt'
