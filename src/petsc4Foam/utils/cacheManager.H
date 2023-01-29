/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2020 OpenCFD Ltd.
    Copyright (C) 2020 Simone Bna
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
    Foam::cacheManager

Description
    The manager class for the caching of matrix and preconditioner.

    An Example is:

    \verbatim
    petsc
    {
        ...
        caching
        {
            matrix
            {
                update  always;  // (default: always)
            }

            preconditioner
            {
                update  periodic;

                periodicCoeffs
                {
                    frequency   3;
                }
            }
        }
    }
    \endverbatim

SourceFiles
    cacheManager.C

\*---------------------------------------------------------------------------*/

#ifndef foamCacheManager_H
#define foamCacheManager_H

#include "Enum.H"
#include "clockValue.H"
#include "dictionary.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace PetscUtils
{

/*---------------------------------------------------------------------------*\
                           Class Caching Declaration
\*---------------------------------------------------------------------------*/

struct Caching
{
    // Public Data Types

        //- Caching update types
        enum updateTypes
        {
            Always,         //!< "always" update every time-step [default]
            Periodic,       //!< "periodic" update at given period
            Adaptive,       //!< "adaptive" update scheme
            Never,          //!< "never" update (or "none")
        };


        //- Names for the update types
        static const Enum<updateTypes> updateTypeNames_;


    // Member Data

        //- Caching update
        updateTypes updateType_;

        //- Cache update frequency (for Periodic)
        int updateFreq_;

        //- Elapsed time (s) for current iteration
        double timeIter;

        //- Elapsed time (s) for first iteration
        double time0;

        //- Elapsed time (s) for second iteration
        double time1;

        //- Timer value
        clockValue timer_;


    // Constructors

        //- Default construct. Always update (no caching).
        Caching()
        :
           updateType_{updateTypes::Always},
           updateFreq_{1},
           timeIter{0},
           time0{0},
           time1{0}
        {}


    void init(const word& key, const dictionary& dict)
    {
        dictionary cacheDict = dict.subOrEmptyDict(key);
        updateType_ =
            updateTypeNames_.getOrDefault
            (
                "update",
                cacheDict,
                updateTypes::Always,
                false  // non-failsafe - Fatal on bad enumeration
            );

        if (updateType_ == updateTypes::Periodic)
        {
            dictionary coeffs(cacheDict.subOrEmptyDict("periodicCoeffs"));
            updateFreq_ = coeffs.getOrDefault<int>("frequency", 1);
        }
    }
};

} // End namespace PetscUtils


/*---------------------------------------------------------------------------*\
                     Class cacheManager Declaration
\*---------------------------------------------------------------------------*/

class cacheManager
{
    // Private Data

        //- The current (relative) iterations for matrix
        label miter;

        //- The current (relative) iterations for preconditioner
        label piter;

        PetscUtils::Caching matrixCaching;
        PetscUtils::Caching precondCaching;


public:

    // Constructors

        //- Default construct
        cacheManager()
        :
            miter{0},
            piter{0}
        {}


    //- Destructor
    ~cacheManager() = default;


    void init(const dictionary& dict)
    {
        matrixCaching.init("matrix", dict);
        precondCaching.init("preconditioner", dict);
    }


    // Member Functions

    bool needsMatrixUpdate()
    {
        return needsUpdate(matrixCaching, miter);
    }

    bool needsPrecondUpdate()
    {
        return needsUpdate(precondCaching, piter);
    }

    void eventBegin()
    {
        if (precondCaching.updateType_ == PetscUtils::Caching::Adaptive)
        {
            // Begin timing interval
            precondCaching.timer_.update();
        }
    }

    void eventEnd()
    {
        if (precondCaching.updateType_ == PetscUtils::Caching::Adaptive)
        {
            // Elapsed timing interval (s)
            precondCaching.timeIter = precondCaching.timer_.elapsed();

            if (piter == 0)
            {
                precondCaching.time0 = precondCaching.timeIter;
            }
            else if (piter == 1)
            {
                precondCaching.time1 = precondCaching.timeIter;
            }
        }

        if
        (
            precondCaching.updateType_ == PetscUtils::Caching::Periodic
         || precondCaching.updateType_ == PetscUtils::Caching::Adaptive
        )
        {
            ++piter;
        }
        if
        (
            matrixCaching.updateType_ == PetscUtils::Caching::Periodic
        )
        {
            ++miter;
        }
    }


private:

    bool needsUpdate(const PetscUtils::Caching& caching, label& iter) const
    {
        // Default: Always update
        bool need = true;

        switch (caching.updateType_)
        {
            case PetscUtils::Caching::Never:
            {
                need = false;
                break;
            }

            case PetscUtils::Caching::Always:
            {
                break;
            }

            case PetscUtils::Caching::Periodic:
            {
                if
                (
                    caching.updateFreq_ <= 1
                 || (iter % caching.updateFreq_) == 0
                )
                {
                    iter = 0;
                }
                else
                {
                    need = false;
                }
                break;
            }

            case PetscUtils::Caching::Adaptive:
            {
                if (iter > 3) // we need at least three times
                {
                    const double ratio0 =
                        caching.time0 / caching.timeIter;

                    const double ratio1 =
                        caching.time1 / caching.timeIter;

                    const int nsteps =
                        min(1e5, ratio0 * (1. / mag(1. - ratio1 + 1e-6)));

                    if (iter >= nsteps)
                    {
                        iter = 0;
                    }
                    else
                    {
                        need = false;
                    }
                }
                break;
            }
        }

        return need;
    }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
