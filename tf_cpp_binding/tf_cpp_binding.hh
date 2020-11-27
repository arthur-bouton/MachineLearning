/* 
** C++ binding of TensorFlow native C API to import a model and feed and
** retrieve unidimensional vectors.
**
** You can check the input names of the model by executing in a terminal:
**
** $ saved_model_cli show --dir <path_to_model_dir> --tag_set serve --signature_def serving_default
**
**
** Copyright (C) 2020 Arthur BOUTON
** 
** This program is free software: you can redistribute it and/or modify  
** it under the terms of the GNU General Public License as published by  
** the Free Software Foundation, version 3.
**
** This program is distributed in the hope that it will be useful, but 
** WITHOUT ANY WARRANTY; without even the implied warranty of 
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
** General Public License for more details.
**
** You should have received a copy of the GNU General Public License 
** along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef TF_CPP_BINDING_HH
#define TF_CPP_BINDING_HH

#include "tensorflow/c/c_api.h"
#include <memory>
#include <vector>


//------------------------------------------------------------------------------------------//
// Template class to import a tensorflow model and feed and retrieve unidimensional vectors //
//------------------------------------------------------------------------------------------//
template <class T>
class TF_model
{
	public:

	typedef std::shared_ptr<TF_model> ptr_t;

	typedef std::vector<std::vector<T>> vector_set_t;

	// dim_inputs and dim_outputs list the dimensions of each expected input and output vectors:
	TF_model( const char* path_to_model_dir, const std::vector<int>& dim_inputs,
	                                         const std::vector<int>& dim_ouputs );

	// Infer the output(s) one set of input(s) at a time:
	vector_set_t infer( vector_set_t inputs );

	~TF_model();

	protected:

	int _n_inputs, _n_outputs;
	std::vector<int> _dim_inputs, _dim_outputs;
	TF_Graph* _graph;
	TF_Status* _status;
	TF_SessionOptions* _sessionOpts;
	TF_Session* _session;
	TF_Output* _model_inputs;
	TF_Output* _model_outputs;
	TF_Tensor** _input_tensors;
	TF_Tensor** _output_tensors;
};


// Declaration of an alias declaration following the C++11 standard:
template <class T>
using ptr_t = std::shared_ptr<TF_model<T>>;


#endif
