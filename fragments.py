def condentropy_2vars(self, which_sources):
    # compute cond entropy of the distribution in self.sol_rpq
    mysum = 0.
    if which_sources == [1,2]:
        # H( S | X, Y )
        # Fix x,y
        marg_XY = self.marginal_ab(self.X, self.Y, self.S, self.Z, which_sources)
        marg_SXY = self.marginal_abc(self.S, self.X, self.Y, self.Z, which_sources)
        print(marg_SXY)
        for s in self.S:
            for x in self.X:
                for y in self.Y:
                    if (s,x,y) in self.idx_of_trip.keys():
                        # subtract q_{sxy}*log( q_{sxy}/q_{xy} )
                        mysum -= marg_SXY[(s,x,y)]*log(marg_SXY[(s,x,y)]/marg_XY[(x,y)])
        return mysum
    #^ if sources
    elif which_sources == [1,3]:
        # H( S | X, Z )
        # Fix x,z
        for x in self.X:
            for z in self.Z:
                marg_xz = 0.
                q_list = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for s in self.S for y in self.Y if (s,x,y,z) in self.idx_of_quad.keys()]
                print("q_list: ", q_list)
                # Compute q_{xz} 
                for i in q_list: marg_xz += max(0,self.sol_rpq[i])
                #^ for i
                # Fix s
                for s in self.S:
                    marg_sxz = 0.
                    q_list_ = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for y in self.Y if (s,x,y,z) in self.idx_of_quad.keys()]
                    print("q_list_", q_list_)                    
                    # Compute q_{sxz}
                    for i in q_list_: marg_sxz += max(0, self.sol_rpq[i])
                    #^ for i
                    
                    # subtract q_{sxz}*log( q_{sxz}/q_{xz} )
                    if marg_sxz > 0: mysum -= marg_sxz*log(marg_sxz/marg_xz)
                #^ for s 
            #^ for z
        #^ for x
        return mysum
    #^ if sources
    elif which_sources == [2,3]:
        # H( S | Y, Z )
        # Fix y,z
        for y in self.Y:
            for z in self.Z: 
                marg_yz = 0.
                q_list = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for s in self.S for x in self.X if (s,x,y,z) in self.idx_of_quad.keys()]
                print("q_list: ", q_list)                
                # Compute q_{yz}
                for i in q_list: marg_yz += max(0,self.sol_rpq[i])
                #^ for i
                
                # Fix s
                for s in self.S:
                    marg_syz = 0.
                    q_list_ = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for x in self.X if (s,x,y,z) in self.idx_of_quad.keys()]
                    print("q_list_", q_list_)                    
                    # Compute q_{syz}
                    for i in q_list_: marg_syz += max(0, self.sol_rpq[i])
                    #^ for i
                    
                    # subtract q_{syz}*log( q_{syz}/q_{yz} )  
                    if marg_syz > 0 :mysum -= marg_syz*log(marg_syz/marg_yz)
                #^ for s 
            #^ for z
        #^ for y
        return mysum
    #^ if sources
#^ condentropy_2vars()

def condentropy_1var(self,which_sources):
    mysum = 0. 
    if which_sources == [1,2]:
        # H( S | Z )
        # Fix z 
        for z in self.Z:
            marg_z = 0.
            q_list = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for s in self.S for x in self.X for y in self.Y if (s,x,y,z) in self.idx_of_quad.keys()]

            # Compute q_{z}
            for i in q_list: marg_z += max(0,self.sol_rpq[i])
            #^ for i

            # Fix s
            for s in self.S:
                marg_sz = 0.
                q_list_ = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for x in self.X for y in self.Y if (s,x,y,z) in self.idx_of_quad.keys()]

                # Compute q_{s,z}
                for i in q_list_: marg_sz += max(0, self.sol_rpq[i])
                #^ for i
            
                # Subtract q_{s,z}*log( q_{s,z}/ q_{z} )
                if marg_sz > 0: mysum -= marg_sz*log(marg_sz/marg_z)
            #^ for s 
        #^ for z
        return mysum
    #^ if sources
    elif which_sources == [1,3]:
        # H( S | Y ) 
        # Fix y 
        for y in self.Y:
            marg_y = 0.
            q_list = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for s in self.S for x in self.X for z in self.Z if (s,x,y,z) in self.idx_of_quad.keys()]

            # Compute q_{y}
            for i in q_list: marg_y += max(0,self.sol_rpq[i])
            #^ for i

            # Fix s
            for s in self.S:
                marg_sy = 0.
                q_list_ = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for x in self.X for z in self.Z if (s,x,y,z) in self.idx_of_quad.keys()]

                # Compute q_{s,y}
                for i in q_list_: marg_sy += max(0, self.sol_rpq[i])
                #^ for i
            
                # Subtract q_{s,y}*log( q_{s,y}/ q_{y} )
                if marg_sy > 0: mysum -= marg_sy*log(marg_sy/marg_y)
            #^ for s 
        #^ for y
        return mysum
    #^ if sources
    
    elif which_sources == [2,3]:
        # H ( S | X )
        # Fix x 
        for x in self.X:
            marg_x = 0.
            q_list = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for s in self.S for y in self.Y for z in self.Z if (s,x,y,z) in self.idx_of_quad.keys()]

            # Compute q_{x}
            for i in q_list: marg_x += max(0,self.sol_rpq[i])
            #^ for i

            # Fix s
            for s in self.S:
                marg_sx = 0.
                q_list_ = [ self.sq_vidx(self.idx_of_quad[ (s,x,y,z) ], which_sources) for y in self.Y for z in self.Z if (s,x,y,z) in self.idx_of_quad.keys()]

                # Compute q_{s,x}
                for i in q_list_: marg_sx += max(0, self.sol_rpq[i])
                #^ for i
            
                # Subtract q_{s,x}*log( q_{s,x}/ q_{x} )
                if marg_sx > 0: mysum -= marg_sx*log(marg_sx/marg_x)
            #^ for s 
        #^ for x
        return mysum
    #^ if sources
#^ condentropy_1var()
