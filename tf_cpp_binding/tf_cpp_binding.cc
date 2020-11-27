#include "tf_cpp_wrapper.hh"
#include <stdexcept>
#include <string>


template class TF_model<float>;
template class TF_model<double>;


void NoOpDeallocator( void* data, size_t a, void* b ) {}


template <class T>
TF_model<T>::TF_model( const char* path_to_model_dir, const std::vector<int>& dim_inputs,
										              const std::vector<int>& dim_outputs ) :
													  _n_inputs( dim_inputs.size() ),
													  _n_outputs( dim_outputs.size() ),
													  _dim_inputs( dim_inputs ),
													  _dim_outputs( dim_outputs )
{
	// Import the model:
	_graph = TF_NewGraph();
	_status = TF_NewStatus();
	_sessionOpts = TF_NewSessionOptions();

	int ntags = 1;
	const char* tags = "serve"; 

	_session = TF_LoadSessionFromSavedModel( _sessionOpts, nullptr, path_to_model_dir, &tags, ntags, _graph, nullptr, _status );

	if( TF_GetCode( _status ) != TF_OK )
		throw std::runtime_error( "Failed to create the TensorFlow session: " + std::string( TF_Message( _status ) ) );


	// Retrieve the input(s):
	_model_inputs = (TF_Output*) malloc( sizeof( TF_Output )*_n_inputs );
	for ( int i = 0 ; i < _n_inputs ; i++ )
	{
		std::string input_name = "serving_default_input_" + std::to_string( i + 1 );

		TF_Output endpoint = { TF_GraphOperationByName( _graph, input_name.c_str() ), 0 };

		if ( endpoint.oper == nullptr )
			throw std::runtime_error( "Failed to retrieve the input " + input_name );

		_model_inputs[i] = endpoint;
	}

	// Retrieve the output(s):
	const char* output_name = "StatefulPartitionedCall";

	_model_outputs = (TF_Output*) malloc( sizeof( TF_Output )*_n_outputs );
	for ( int i = 0 ; i < _n_outputs ; i++ )
	{
		TF_Output endpoint = { TF_GraphOperationByName( _graph, output_name ), i };

		if ( endpoint.oper == nullptr )
			throw std::runtime_error( "Failed to retrieve the output " + std::string( output_name ) );

		_model_outputs[i] = endpoint;
	}

	// Allocate in the memory for the input and output data:
	_input_tensors  = (TF_Tensor**) malloc( sizeof( TF_Tensor* )*_n_inputs );
	_output_tensors = (TF_Tensor**) malloc( sizeof( TF_Tensor* )*_n_outputs );
}


template <class T>
typename TF_model<T>::vector_set_t TF_model<T>::infer( typename TF_model<T>::vector_set_t inputs )
{
	// Build the input tensor:
	for ( int i = 0 ; i < _n_inputs ; i++ )
	{
		int ndims = 2;
		int64_t dims[] = { 1, _dim_inputs[i] };
		int ndata = sizeof( T )*_dim_inputs[i];
		T* data = inputs[i].data();

		TF_DataType datatype;
		if ( std::is_same<T,float>::value )
			datatype = TF_FLOAT;
		else if ( std::is_same<T,double>::value )
			datatype = TF_DOUBLE;
		else
			throw std::runtime_error( "This type hasn't been implemented" );

		_input_tensors[i] = TF_NewTensor( datatype, dims, ndims, data, ndata, &NoOpDeallocator, 0 );

		if ( _input_tensors[i] == nullptr )
			throw std::runtime_error( "Failed to create the input tensor " + std::to_string( i + 1 ) );
	}

	// Run the Session:
	TF_SessionRun( _session, nullptr, _model_inputs, _input_tensors, _n_inputs, _model_outputs, _output_tensors, _n_outputs, nullptr, 0, nullptr , _status );

	for ( int i = 0 ; i < _n_inputs ; i++ )
		TF_DeleteTensor( _input_tensors[i] );

	if( TF_GetCode( _status ) != TF_OK )
		throw std::runtime_error( "Failed to run the session: " + std::string( TF_Message( _status ) ) );

	// Format the outputs:
	vector_set_t outputs;
	for ( int i = 0 ; i < _n_outputs ; i++ )
	{
		T* values = (T*) TF_TensorData( _output_tensors[i] );
		std::vector<T> vector( values, values + _dim_outputs[i] );
		outputs.push_back( vector );
	}

	return outputs;
}


template <class T>
TF_model<T>::~TF_model()
{
	TF_DeleteGraph( _graph );
	TF_DeleteSession( _session, _status );
	TF_DeleteSessionOptions( _sessionOpts );
	TF_DeleteStatus( _status );
	free( _model_inputs );
	free( _model_outputs );
	free( _input_tensors );
	free( _output_tensors );
}
