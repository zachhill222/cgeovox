#pragma once

namespace gv::pde
{
	//Base class for PDE problems
	//The problem is allowed to make changes to the mesh but dofhandlers are not
	//any mesh refinement/unrefinement should probably go
	// element error indicator -> dofrefine (stage + tell the mesh which elements to refine) -> mesh.processSplit -> dofprocess_refine

	
}