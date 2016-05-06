load('Z:\tzeng\bigneuron\big_neuron_hackthon_tesla\data\new_train.mat')
d=data{1};
l=elm_labels{1};
d=single(d);
l=single(l);
%l=permute(l,[3 2 1]);
%d=permute(d,[3 2 1]);
d_size=size(d);
%s1=[[1,1] d_size ];
%d_x=reshape(d,s1);
%l_x=reshape(l,s1);
%d_x=permute(d_x,[5 4 3 2 1]);
%l_x=permute(l_x,[5 4 3 2 1]);
d_details.location = '/';
d_details.Name = 'data';
l_details.location = '/';
l_details.Name = 'label';
hdf5write('Z:\tzeng\bigneuron\big_neuron_hackthon_tesla\data\hd5_train.h5',d_details,d,l_details,l)