/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::petscLinearSolverContext

Description
    A struct containing a KSP and Mat for each equation to be solved

SourceFiles
    petscLinearSolverContext.C

\*---------------------------------------------------------------------------*/

#ifndef petscFoamLinearSolverContext_H
#define petscFoamLinearSolverContext_H

#include "linearSolverContext.H"
#include "petscUtils.H"

// PETSc
#include "petscvec.h"
#include "petscmat.h"
#include "petscksp.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class petscLinearSolverContext Declaration
\*---------------------------------------------------------------------------*/

class petscLinearSolverContext
:
    public linearSolverContext
{
    // Private Data

        bool init_aux_vectors_;

public:

    //- Runtime information
        TypeName("petscLinearSolverContext");

    // Public Members

        Mat Amat;
        KSP ksp;

        // Auxiliary vectors for the OpenFOAM-PETSc L1-norm
        Vec ArowsSum;
        Vec *res_l1_w;

        List<PetscInt> lowNonZero;
        label maxLowNonZeroPerRow;

        PetscScalar normFactor;

        PetscLogStage matstage;
        PetscLogStage pcstage;
        PetscLogStage kspstage;

        static const word packageName;

    // Constructors

        //- Default construct
        petscLinearSolverContext(const word eqName, const fileName& optionsFile)
        :
            linearSolverContext(eqName),
            init_aux_vectors_(false),
            Amat(nullptr),
            ksp(nullptr),
            ArowsSum(nullptr),
            res_l1_w(nullptr)
        {}


    //- Destructor
    virtual ~petscLinearSolverContext()
    {
        MatDestroy(&Amat);
        KSPDestroy(&ksp);
        VecDestroy(&ArowsSum);
        if (res_l1_w) VecDestroyVecs(2,&res_l1_w);
    }


    // Member Functions

        //- Create auxiliary rows for calculation purposes
        void createAuxVecs(Vec psi)
        {
            if (!init_aux_vectors_)
            {
                init_aux_vectors_ = true;
                VecDuplicate(psi, &ArowsSum);
                VecDuplicateVecs(psi, 2, &res_l1_w);
            }
        }

        //- Sum the rows of A, placing the result in ArowsSum
        void initArowsSumVec(Mat A)
        {
            if (init_aux_vectors_)
            {
                MatGetRowSum(A, ArowsSum);
            }
        }

        //- Compute the normFactor used in convergence testing,
        //- assumes ArowsSum has been already computed
        void computeNormFactor(Mat A, Vec psi, Vec source)
        {
            if (!init_aux_vectors_) return;

            MatMult(A, psi, res_l1_w[1]);

            normFactor =
               Foam::PetscUtils::normFactor
               (
                   res_l1_w[1],
                   psi,
                   source,
                   ArowsSum
               );
        }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
