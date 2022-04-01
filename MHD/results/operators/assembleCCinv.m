function CCi = assembleCCinv( precType, Fa, MaNZ, Wa, dtuWa, Ka, B0, dt, essA )
%assembleCCinv Provides various ways of defining the space-time wave equation appearing in the magnetic Schur complement approximation
%
% Syntax:  CCi = assembleCCinv( precType, Fa, MaNZ, Wa, dtuWa, Ka, B0, dt, essA )
% 
%
%
% Author: Federico Danieli, Numerical Analysis Group
% University of Oxford, Dept. of Mathematics
% email address: federico.danieli@maths.ox.ac.uk  
% Mar 2021; Last revision: Mar-2021



NT = length(Fa);

CCpinv = cell(1,NT);
CC0    = cell(1,NT);
CCm    = cell(1,NT);


switch precType
  %% These operators directly translate the structure of the single time-step precon to the space-time case
  % Full Fa Ma^-1 Fa + |B0|Ka operator
  case 0
    MaNZinv = inv(full(MaNZ));
    for pp = 1:NT
      temp = Fa{pp}*MaNZinv*Fa{pp} + dt*dt*B0{pp}*Ka;
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');
      if pp < NT
        CC0{pp} = - (Fa{pp}+Fa{pp+1});
        CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
        CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));
      end
      CCm{pp} =  MaNZ;
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % Full Fa Ma^-1 Fa + |B0|Ka operator, with inverse of mass matrix approximated with its diagonal
  case 1
    MaNZinv = spdiags(1./diag(MaNZ),0,size(MaNZ,1),size(MaNZ,2));
    for pp = 1:NT
      temp = Fa{pp}*MaNZinv*Fa{pp} + dt*dt*B0{pp}*Ka;
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');
      if pp < NT
        CC0{pp} = - (Fa{pp}+Fa{pp+1});
        CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
        CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));
      end
      CCm{pp} =  MaNZ;
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % -- Terms in dt coming from space-time Fa*Mai*Fa + |B0|Ka operator
  case 2
    for pp = 1:NT
      temp = MaNZ + 2*dt*Wa{pp} + dt*dt*B0{pp}*Ka;
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');

      if pp < NT
        CC0{pp} =  -2*MaNZ - dt*(Wa{pp}+Wa{pp+1});
        CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
        CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));
      end
      CCm{pp} =  MaNZ; % maybe "symmetrise" this?
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end




  %% These operators take into account the attempt to discretise a second order temporal derivative
  % Full Fa Ma^-1 Fa + |B0|Ka operator
  case 10
    MaNZinv = inv(full(MaNZ));
    for pp = 1:NT
      Wa{pp}(essA,:)    = sparse(length(essA),size(Wa{pp},2));
      Wa{pp}(:,essA)    = sparse(size(Wa{pp},1),length(essA));

      temp    =    MaNZ + dt*Wa{pp} + 0.25*dt*dt*( Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ             + 0.5*dt*dt*(  Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ - dt*Wa{pp} + 0.25*dt*dt*( Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % Full Fa Ma^-1 Fa + |B0|Ka operator, with inverse of mass matrix approximated with its diagonal
  case 11
    MaNZinv = spdiags(1./diag(MaNZ),0,size(MaNZ,1),size(MaNZ,2));
    for pp = 1:NT
      Wa{pp}(essA,:)    = sparse(length(essA),size(Wa{pp},2));
      Wa{pp}(:,essA)    = sparse(size(Wa{pp},1),length(essA));

      temp    =    MaNZ + dt*Wa{pp} + 0.25*dt*dt*( Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ             + 0.5*dt*dt*(  Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ - dt*Wa{pp} + 0.25*dt*dt*( Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % -- Terms in dt coming from space-time Fa*Mai*Fa + |B0|Ka operator
  case 12
    for pp = 1:NT
      temp    =    MaNZ + dt*Wa{pp} + 0.25*dt*dt*( B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ             + 0.5*dt*dt*(  B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ - dt*Wa{pp} + 0.25*dt*dt*( B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % Pure wave equation with implicit leapfrog discretisation
  case 13
    for pp=1:NT
      temp    =    MaNZ +           + 0.25*dt*dt*( B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ             + 0.5*dt*dt*(  B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ             + 0.25*dt*dt*( B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % -- ALL terms in dt coming from continuous counterpart of space-time Fa*Mai*Fa + |B0|Ka operator
  case 14
    for pp = 1:NT
      temp    =    MaNZ + dt*(Wa{pp}-dtuWa{pp}) + 0.25*dt*dt*( B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ                         + 0.5*dt*dt*(  B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ - dt*(Wa{pp}-dtuWa{pp}) + 0.25*dt*dt*( B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end



  %% These operators adjust the "continuous" interpretation so that it matches the main diagonal in the purely discrete ones
  % Full Fa Ma^-1 Fa + |B0|Ka operator
  case 20
    MaNZinv = inv(full(MaNZ));
    for pp = 1:NT
      Wa{pp}(essA,:)    = sparse(length(essA),size(Wa{pp},2));
      Wa{pp}(:,essA)    = sparse(size(Wa{pp},1),length(essA));

      temp    =    MaNZ + 2*dt*Wa{pp} +     dt*dt*( Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ               +   2*dt*dt*(  Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ - 2*dt*Wa{pp} +     dt*dt*( Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % Full Fa Ma^-1 Fa + |B0|Ka operator, with inverse of mass matrix approximated with its diagonal
  case 21
    MaNZinv = spdiags(1./diag(MaNZ),0,size(MaNZ,1),size(MaNZ,2));
    for pp = 1:NT
      Wa{pp}(essA,:)    = sparse(length(essA),size(Wa{pp},2));
      Wa{pp}(:,essA)    = sparse(size(Wa{pp},1),length(essA));

      temp    =    MaNZ + 2*dt*Wa{pp} +     dt*dt*( Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ               +   2*dt*dt*(  Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ - 2*dt*Wa{pp} +     dt*dt*( Wa{pp}*MaNZinv*Wa{pp} + B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % -- Terms in dt coming from space-time Fa*Mai*Fa + |B0|Ka operator
  case 22
    for pp = 1:NT
      temp    =    MaNZ + 2*dt*Wa{pp} +     dt*dt*( B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA)); 
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ               +   2*dt*dt*(  B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ - 2*dt*Wa{pp} +     dt*dt*( B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % Pure wave equation with implicit leapfrog discretisation
  case 23
    for pp=1:NT
      temp    =    MaNZ +           +     dt*dt*( B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA)); 
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ             +   4*dt*dt*(  B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ             +     dt*dt*( B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end
  % -- ALL terms in dt coming from continuous counterpart of space-time Fa*Mai*Fa + |B0|Ka operator
  case 24
    for pp = 1:NT
      temp    =    MaNZ + 2*dt*(Wa{pp}-dtuWa{pp}) +     dt*dt*( B0{pp}*Ka );
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA)); 
      CCpinv{pp} = decomposition(temp, 'lu');
  
      CCm{pp} = -2*MaNZ                           +   2*dt*dt*(  B0{pp}*Ka );
      CC0{pp}(essA,:) = sparse(length(essA),size(CC0{pp},2));
      CC0{pp}(:,essA) = sparse(size(CC0{pp},1),length(essA));

      CCm{pp} =    MaNZ - 2*dt*(Wa{pp}-dtuWa{pp}) +     dt*dt*( B0{pp}*Ka );
      CCm{pp}(essA,:) = sparse(length(essA),size(CCm{pp},2));
      CCm{pp}(:,essA) = sparse(size(CCm{pp},1),length(essA));
    end



  %% These operators ignore the space-time structure, and just consider the main diagonal
  % Full Fa Ma^-1 Fa + |B0|Ka operator
  case 30
    MaNZinv = inv(full(MaNZ));
    for pp = 1:NT
      temp = Fa{pp}*MaNZinv*Fa{pp} + dt*dt*B0{pp}*Ka;
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');

      CC0{pp} = sparse(size(MaNZ,1),size(MaNZ,2));
      CCm{pp} = sparse(size(MaNZ,1),size(MaNZ,2));
    end
  % Full Fa Ma^-1 Fa + |B0|Ka operator, with inverse of mass matrix approximated with its diagonal
  case 31
    MaNZinv = spdiags(1./diag(MaNZ),0,size(MaNZ,1),size(MaNZ,2));
    for pp = 1:NT
      temp = Fa{pp}*MaNZinv*Fa{pp} + dt*dt*B0{pp}*Ka;
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');

      CC0{pp} = sparse(size(MaNZ,1),size(MaNZ,2));
      CCm{pp} = sparse(size(MaNZ,1),size(MaNZ,2));
    end
  % -- Terms in dt coming from space-time Fa*Mai*Fa + |B0|Ka operator
  case 32
    for pp = 1:NT
      temp = MaNZ + 2*dt*Wa{pp} + dt*dt*B0{pp}*Ka;
      temp(essA,:)    = sparse(length(essA),size(temp,2));
      temp(:,essA)    = sparse(size(temp,1),length(essA));
      temp(essA,essA) = eye(length(essA));
      CCpinv{pp} = decomposition(temp, 'lu');

      CC0{pp} = sparse(size(MaNZ,1),size(MaNZ,2));
      CCm{pp} = sparse(size(MaNZ,1),size(MaNZ,2));
    end



  otherwise
    error('Type of solver for space-time wave equation not recognised');
end

if (precType<30)
  CCi  = @(x) invertBlockLowTri(x,{CCpinv,CC0,CCm},essA);
else
  CCi  = @(x) invertBlockLowTri(x,{CCpinv},essA);
end









  %   CCpinv = cell(1,NT);
  % CC0    = cell(1,NT);
  % CCm    = cell(1,NT);
  % for pp = 1:NT   % first of all, kill all contributions to Dirichlet nodes
  %   Wa{pp}(essA,:) = sparse(length(essA),size(Wa{1},2));
  %   Wa{pp}(:,essA) = sparse(size(Wa{1},1),length(essA));
  % end
  % switch magSchurComp
  %   % -- Inverse of whole Fa*Mai*Fa = Fai*Ma*Fai  -- HOLDS ONLY IF |B0|=0!!
  %   case 0
  %     Fainv = cell(1,NT);
  %     for pp=1:NT
  %       Fainv{pp} = decomposition(Fa{pp}, 'lu');
  %     end
  %     FFai = @(x) invertBlockLowTri(x,{Fainv,{Ma}},essA);
  %     CCi  = @(x) FFai( MMa*FFai(x) );
  %   % -- Pure wave equation:
  %   case 1
  %     for pp=1:NT
  %       if pp == 1
  %         CCpinv{pp} = decomposition(Cp{pp}, 'chol');  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp
  %       else
  %         CCpinv{pp} = decomposition(Cp{pp}, 'chol');
  %       end
  %       CC0{pp} = C0{pp};
  %       CCm{pp} = Cm{pp};
  %     end
  %     CCi  = @(x) invertBlockLowTri(x,{CCpinv,CC0,CCm},essA);
  %   % -- Wave equation + terms in dt coming from space-time Fa*Mai*Fa
  %   case 2
  %     for pp = 1:NT
  %       if pp == 1
  %         temp = Cp{pp} + 2*dt*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       else
  %         temp =   Cp{pp} + 2*dt*Wa{pp};
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       end
  %       if pp < NT
  %         CC0{pp} =  C0{pp} -   dt*(Wa{pp}+Wa{pp+1});
  %       end
  %       CCm{pp} =  Cm{pp}; % maybe "symmetrise" this?
  %     end
  %     CCi  = @(x) invertBlockLowTri(x,{CCpinv,CC0,CCm},essA);
  %   % -- Wave equation + terms in dt, but organised as if it was the discretisation of a dissipative wave eq
  %   case 3
  %     for pp = 1:NT
  %       if pp == 1
  %         temp = Cp{pp} + 2*dt*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       else
  %         temp =   Cp{pp} + 2*dt*Wa{pp};
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       end
  %       CC0{pp} =  C0{pp};
  %       CCm{pp} =  Cm{pp} - 2*dt*Wa{pp};
  %     end
  %     CCi  = @(x) invertBlockLowTri(x,{CCpinv,CC0,CCm},essA);
  %   % -- Wave equation + *ALL* terms in dt which would come from the continuous operator: also dtu gradA
  %   case 4
  %     for pp = 1:NT
  %       temp =   Cp{pp} + 2*dt*Wa{pp} + dt*dtuWa{pp}/4;
  %       CCpinv{pp} = decomposition(temp, 'lu');
  %       CC0{pp} =  C0{pp} + 2*dt*dtuWa{pp}/4;
  %       CCm{pp} =  Cm{pp} - 2*dt*Wa{pp} + dt*dtuWa{pp}/4;
  %     end
  %     CCi  = @(x) invertBlockLowTri(x,{CCpinv,CC0,CCm},essA);
  %   % -- Wave equation + terms in dt and terms in dt^2 with inverse of diagonal of mass matrix
  %   case 5
  %     for pp = 1:NT
  %       if pp == 1
  %         temp = Cp{pp} + 2*dt*Wa{pp} + dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       else
  %         temp =   Cp{pp} + 2*dt*Wa{pp} + dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       end
  %       if pp < NT
  %         CC0{pp} =  C0{pp} -   dt*(Wa{pp}+Wa{pp+1});
  %       end
  %       CCm{pp} =  Cm{pp}; % maybe "symmetrise" this?
  %     end
  %     CCi  = @(x) invertBlockLowTri(x,{CCpinv,CC0,CCm},essA);
  %   % -- Wave equation + terms in dt and dt^2, but as if they came from discretising a continuous operator
  %   case 6
  %     for pp = 1:NT
  %       temp =      Cp{pp} + 2*dt*Wa{pp} + 4*dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp}/4;  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
  %       CCpinv{pp} = decomposition(temp, 'lu');
  %       if pp < NT
  %         CC0{pp} = C0{pp}               + 8*dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp}/4;
  %       end
  %       CCm{pp} =   Cm{pp} - 2*dt*Wa{pp} + 4*dt*dt*Wa{pp}*spdiags(1./diag(MaNZ),0,NA,NA)*Wa{pp}/4;
  %     end
  %     CCi  = @(x) invertBlockLowTri(x,{CCpinv,CC0,CCm},essA);
  %   % -- Wave equation + terms in dt and terms in dt^2 with inverse of full mass matrix
  %   case 7
  %     for pp = 1:NT
  %       if pp == 1
  %         temp = Cp{pp} + 2*dt*Wa{pp} + dt*dt*Wa{pp}*inv(MaNZ)*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       else
  %         temp = Cp{pp} + 2*dt*Wa{pp} + dt*dt*Wa{pp}*inv(MaNZ)*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       end
  %       if pp < NT
  %         CC0{pp} =  C0{pp} -   dt*(Wa{pp}+Wa{pp+1});
  %       end
  %       CCm{pp} =  Cm{pp}; % maybe "symmetrise" this?
  %     end
  %     CCi  = @(x) invertBlockLowTri(x,{CCpinv,CC0,CCm},essA);
  %   % -- Wave equation + terms in dt and terms in dt^2 with inverse of full mass matrix, but consider only diagonal
  %   case 8
  %     for pp = 1:NT
  %       if pp == 1
  %         temp = Cp{pp} + 2*dt*Wa{pp} + dt*dt*Wa{pp}*inv(MaNZ)*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       else
  %         temp = Cp{pp} + 2*dt*Wa{pp} + dt*dt*Wa{pp}*inv(MaNZ)*Wa{pp};  % non-time-dep & sym operator: rank 0 must solve for Cp+Cm = 2Cp?????
  %         CCpinv{pp} = decomposition(temp, 'lu');
  %       end
  %       if pp < NT
  %         CC0{pp} =  sparse(size(Cp{pp},1),size(Cp{pp},2));
  %       end
  %       CCm{pp} =  sparse(size(Cp{pp},1),size(Cp{pp},2));
  %     end
  %     CCi  = @(x) invertBlockLowTri(x,{CCpinv,CC0,CCm},essA);
  % end