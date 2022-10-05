function tTV_update = compute_tTV_circ(image,weight,beta_square)
%tTV_update = compute_tTV_yt(image,weight,beta_square)
if weight~=0
    image = cat(3,image(:,:,end,:,:,:),image);
    tTV_update = diff(image,1,3);
    tTV_update = tTV_update./(sqrt(beta_square+(abs(tTV_update).^2)));
    tTV_update = cat(3,tTV_update(:,:,end,:,:,:),tTV_update);
    tTV_update = diff(tTV_update,1,3);
    tTV_update = weight .* tTV_update;
    tTV_update = circshift(tTV_update,-1,3);
else
    tTV_update = 0;
end

end