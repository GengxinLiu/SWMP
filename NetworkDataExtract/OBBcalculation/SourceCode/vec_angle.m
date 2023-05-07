
function ang = vec_angle(vec1, vec2)

%ang = atan2(norm(cross(vec1,vec2,2)),dot(vec1,vec2,2));

costheta = dot(vec1,vec2)/(norm(vec1)*norm(vec2));
if(costheta > 1)
    costheta = 1;
end
ang = acos(costheta);

end
