/* -*- Mode: C; indent-tabs-mode: nil; c-basic-offset: 4; tab-width: 4 -*- */

#ifndef TRAX_H
#define TRAX_H

#include <stdio.h>
#include <fcntl.h>

#ifdef TRAX_STATIC_DEFINE
#  define __TRAX_EXPORT
#else
#  ifndef __TRAX_EXPORT
#    if defined(_MSC_VER)
#      ifdef trax_EXPORTS
         /* We are building this library */
#        define __TRAX_EXPORT __declspec(dllexport)
#      else
         /* We are using this library */
#        define __TRAX_EXPORT __declspec(dllimport)
#      endif
#    elif defined(__GNUC__)
#      ifdef trax_EXPORTS
         /* We are building this library */
#        define __TRAX_EXPORT __attribute__((visibility("default")))
#      else
         /* We are using this library */
#        define __TRAX_EXPORT __attribute__((visibility("default")))
#      endif
#    endif
#  endif
#endif

#if defined(__OS2__) || defined(__WINDOWS__) || defined(WIN32) || defined(WIN64) || defined(_MSC_VER)
#define TRAX_NO_LOG (~0)
#if defined(_MSC_VER)
#pragma comment(lib, "ws2_32.lib")
#endif
#else
#define TRAX_NO_LOG -1
#endif

#define TRAX_VERSION 4

#define TRAX_ERROR -1
#define TRAX_OK 0
#define TRAX_HELLO 1
#define TRAX_INITIALIZE 2
#define TRAX_FRAME 3
#define TRAX_QUIT 4
#define TRAX_STATE 5

#define TRAX_IMAGE_EMPTY 0
#define TRAX_IMAGE_PATH 1
#define TRAX_IMAGE_URL 2
#define TRAX_IMAGE_MEMORY 4
#define TRAX_IMAGE_BUFFER 8

#define TRAX_IMAGE_ANY (TRAX_IMAGE_PATH | TRAX_IMAGE_URL | TRAX_IMAGE_MEMORY | TRAX_IMAGE_BUFFER)

#define TRAX_IMAGE_BUFFER_ILLEGAL 0
#define TRAX_IMAGE_BUFFER_PNG 1
#define TRAX_IMAGE_BUFFER_JPEG 2

#define TRAX_IMAGE_MEMORY_ILLEGAL 0
#define TRAX_IMAGE_MEMORY_GRAY8 1
#define TRAX_IMAGE_MEMORY_GRAY16 2
#define TRAX_IMAGE_MEMORY_RGB 3

#define TRAX_REGION_EMPTY 0
#define TRAX_REGION_SPECIAL 1
#define TRAX_REGION_RECTANGLE 2
#define TRAX_REGION_POLYGON 4
#define TRAX_REGION_MASK 8

#define TRAX_REGION_ANY (TRAX_REGION_RECTANGLE | TRAX_REGION_POLYGON | TRAX_REGION_MASK)

#define TRAX_FLAG_VALID 1
#define TRAX_FLAG_SERVER 2
#define TRAX_FLAG_TERMINATED 4

#define TRAX_PARAMETER_VERSION 0
#define TRAX_PARAMETER_CLIENT 1
#define TRAX_PARAMETER_SOCKET 2
#define TRAX_PARAMETER_REGION 3
#define TRAX_PARAMETER_IMAGE 4
#define TRAX_PARAMETER_MULTIOBJECT 5

#define TRAX_CHANNELS 3
#define TRAX_CHANNEL_COLOR 1
#define TRAX_CHANNEL_DEPTH 2
#define TRAX_CHANNEL_IR 4

#define TRAX_CHANNEL_INDEX(I) ( \
    (I) == TRAX_CHANNEL_COLOR ? 0 : ( \
    (I) == TRAX_CHANNEL_DEPTH ? 1 : ( \
    (I) == TRAX_CHANNEL_IR ? 2 : -1)))

#define TRAX_CHANNEL_ID(I) ( \
    (I) == 0 ? TRAX_CHANNEL_COLOR : ( \
    (I) == 1 ? TRAX_CHANNEL_DEPTH : ( \
    (I) == 2 ? TRAX_CHANNEL_IR : -1)))

#define TRAX_LOCALHOST "127.0.0.1"

// Metadata flags
#define TRAX_METADATA_MULTI_OBJECT 1

#ifdef TRAX_LEGACY_SINGLE
#define trax_server_wait trax_server_wait_sot
#define trax_server_reply trax_server_reply_sot
#define trax_server_setup(M, L) trax_server_setup_v(M, L, 3)
#define trax_server_setup_file(M, I, O, L) trax_server_setup_file_v(M, I, O, L, 3)
#else
#define trax_server_wait trax_server_wait_mot
#define trax_server_reply trax_server_reply_mot
#define trax_server_setup(M, L) trax_server_setup_v(M, L, 0)
#define trax_server_setup_file(M, I, O, L) trax_server_setup_file_v(M, I, O, L, 0)
#endif

#define TRAX_SUPPORTS(F, M) (((F) & (M)) != 0)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A trax image data structure. Use trax_image_* functions to access the data.
**/
typedef struct trax_image {
    short type;
    int width;
    int height;
    int format;
    char* data;
} trax_image;

/**
 * A placeholder for region structure. Use the trax_region_* functions to manipulate
 * the data.
**/
typedef void trax_region;

/**
 * A placeholder for properties structure. Use the trax_properties_* functions to manipulate
 * the data.
**/
typedef struct trax_properties trax_properties;

/**
 * A placeholder for object list structure. Use the trax_object_list_* functions to manipulate
 * the data.
**/
typedef struct trax_object_list {
    int size;
    trax_region** regions;
    trax_properties** properties; 
} trax_object_list;

typedef struct trax_bounds {

    float top;
    float bottom;
    float left;
    float right;

} trax_bounds;

typedef void(*trax_logger)(const char *string, int length, void *obj);

typedef void(*trax_enumerator)(const char *key, const char *value, const void *obj);

/**
 * Some basic configuration data used to set up the server.
**/
typedef struct trax_logging {
    int flags;
    trax_logger callback;
    void* data;
} trax_logging;

/**
 * Metadata used to specify tracker metadata and taxonomy.
**/
typedef struct trax_metadata {
    int format_region;
    int format_image;
    int channels; // Encodes the number of channels (RGB, RGB+D, RGB+IR)
    int flags;
    char* tracker_name;
    char* tracker_description;
    char* tracker_family;
    trax_properties* custom;
} trax_metadata;

typedef trax_metadata trax_configuration;

/**
 * Core object of the protocol. Do not manipulate fields directly.
**/
typedef struct trax_handle {
    int flags;
    int version;
    void* stream;
    trax_logging logging;
    trax_metadata* metadata;
    char* error;
    int objects;
} trax_handle;

/**
 * Array for keeping images (RGB, Depth, IR)
**/
typedef struct trax_image_list {
    trax_image* images[TRAX_CHANNELS];
} trax_image_list;

__TRAX_EXPORT extern const trax_logging trax_no_log;

__TRAX_EXPORT extern const trax_bounds trax_no_bounds;

/**
 * Returns library version.
**/
__TRAX_EXPORT const char* trax_version();

/**
 * Create a tracker metadata structure. The last argument contains metadata flags that influence communication modes.
**/
__TRAX_EXPORT trax_metadata* trax_metadata_create(int region_formats, int image_formats, int channels,
    const char* tracker_name, const char* tracker_description, const char* tracker_family, int flags);

/**
 * Correctly releases a metadata structure.
**/
__TRAX_EXPORT void trax_metadata_release(trax_metadata** metadata);

/**
 * A handy function to initialize a logging configuration structure.
**/
__TRAX_EXPORT trax_logging trax_logger_setup(trax_logger callback, void* data, int flags);

/**
 * A handy function to initialize a logging configuration structure for file logging.
**/
__TRAX_EXPORT trax_logging trax_logger_setup_file(FILE* file);

/**
 * Setups the protocol state object for the client and returns a handle object.
**/
__TRAX_EXPORT trax_handle* trax_client_setup_file(int input, int output, const trax_logging log);

/**
 * Setups the protocol state object for the client and returns a handle object.
**/
__TRAX_EXPORT trax_handle* trax_client_setup_socket(int server, int timeout, const trax_logging log);

/**
 * Waits for a valid protocol message from the server.
**/
__TRAX_EXPORT int trax_client_wait(trax_handle* client, trax_object_list** objects, trax_properties* properties);

/**
 * Sends an initialize message, tracking state should be reset.
**/
__TRAX_EXPORT int trax_client_initialize(trax_handle* client, trax_image_list* image, trax_object_list* objects, trax_properties* properties);

/**
 * Sends a frame message.
**/
__TRAX_EXPORT int trax_client_frame(trax_handle* client, trax_image_list* images, trax_object_list* objects, trax_properties* properties);

/**
 * Setups the protocol for the server side and returns a handle object.
**/
__TRAX_EXPORT trax_handle* trax_server_setup_v(trax_metadata *metadata, const trax_logging log, int version);

/**
 * Setups the protocol for the server side and returns a handle object.
**/
__TRAX_EXPORT trax_handle* trax_server_setup_file_v(trax_metadata *metadata, int input, int output, const trax_logging log, int version);

/**
 * Waits for a valid protocol message from the client. This method can only be used in single-object tracking mode.
**/
__TRAX_EXPORT int trax_server_wait_sot(trax_handle* server, trax_image_list** images, trax_region** region, trax_properties* properties);

/**
 * Sends a status reply to the client. This method can only be used in single-object tracking mode.
**/
__TRAX_EXPORT int trax_server_reply_sot(trax_handle* server, trax_region* region, trax_properties* properties);

/**
 * Waits for a valid protocol message from the client. This method can only be used in multi-object tracking mode.
**/
__TRAX_EXPORT int trax_server_wait_mot(trax_handle* server, trax_image_list** images, trax_object_list** objects, trax_properties* properties);

/**
 * Sends a status reply to the client. This method can only be used in single-object tracking mode.
**/
__TRAX_EXPORT int trax_server_reply_mot(trax_handle* server, trax_object_list* objects);

/**
 * Used in client and server. Closes communication, sends quit message if needed.
**/
__TRAX_EXPORT int trax_terminate(trax_handle* handle, const char* reason);

/**
 * Retrieve last error message encountered by the server or client. Returns NULL if no error occured.
**/
__TRAX_EXPORT const char* trax_get_error(trax_handle* handle);

/**
 * Check if the handle is alive or not. The handle is not alive if it was not initalized correctly or 
 * it was terminated.
**/
__TRAX_EXPORT int trax_is_alive(trax_handle* handle);

/**
 * Used in client and server. Closes communication, sends quit message if needed.
 * Releases the handle structure.
**/
__TRAX_EXPORT int trax_cleanup(trax_handle** handle);

/**
 * Sets the parameter of the client or server instance.
**/
__TRAX_EXPORT int trax_set_parameter(trax_handle* handle, int id, int value);

/**
 * Gets the parameter of the client or server instance.
**/
__TRAX_EXPORT int trax_get_parameter(trax_handle* handle, int id, int* value);

/**
 * Releases image structure, frees allocated memory.
**/
__TRAX_EXPORT void trax_image_release(trax_image** image);

/**
 * Creates a file-system path image description.
**/
__TRAX_EXPORT trax_image* trax_image_create_path(const char* path);

/**
 * Creates a URL path image description.
**/
__TRAX_EXPORT trax_image* trax_image_create_url(const char* url);

/**
 * Creates a raw buffer image description.
**/
__TRAX_EXPORT trax_image* trax_image_create_memory(int width, int height, int format);

/**
 * Creates a file buffer image description.
**/
__TRAX_EXPORT trax_image* trax_image_create_buffer(int length, const char* data);

/**
 * Returns a type of the image handle.
**/
__TRAX_EXPORT int trax_image_get_type(const trax_image* image);

/**
 * Returns a file path from a file-system path image description. This function
 * returns a pointer to the internal data which should not be modified.
**/
__TRAX_EXPORT const char* trax_image_get_path(const trax_image* image);

/**
 * Returns a file path from a URL path image description. This function
 * returns a pointer to the internal data which should not be modified.
**/
__TRAX_EXPORT const char* trax_image_get_url(const trax_image* image);

/**
 * Returns the header data of a memory image.
**/
__TRAX_EXPORT void trax_image_get_memory_header(const trax_image* image, int* width, int* height, int* format);

/**
 * Returns a pointer for a writeable row in a data array of an image.
**/
__TRAX_EXPORT char* trax_image_write_memory_row(trax_image* image, int row);

/**
 * Returns a read-only pointer for a row in a data array of an image.
**/
__TRAX_EXPORT const char* trax_image_get_memory_row(const trax_image* image, int row);

/**
 * Returns a file buffer and its length. This function
 * returns a pointer to the internal data which should not be modified.
**/
__TRAX_EXPORT const char* trax_image_get_buffer(const trax_image* image, int* length, int* format);

/**
 * Releases region structure, frees allocated memory.
**/
__TRAX_EXPORT void trax_region_release(trax_region** region);

/**
 * Returns type identifier of the region object.
**/
__TRAX_EXPORT int trax_region_get_type(const trax_region* region);

/**
 * Creates a special region object. Only one paramter (region code) required.
**/
__TRAX_EXPORT trax_region* trax_region_create_special(int code);

/**
 * Sets the code of a special region.
**/
__TRAX_EXPORT void trax_region_set_special(trax_region* region, int code);

/**
 * Returns a code of a special region object.
**/
__TRAX_EXPORT int trax_region_get_special(const trax_region* region);

/**
 * Creates a rectangle region.
**/
__TRAX_EXPORT trax_region* trax_region_create_rectangle(float x, float y, float width, float height);

/**
 * Sets the coordinates for a rectangle region.
**/
__TRAX_EXPORT void trax_region_set_rectangle(trax_region* region, float x, float y, float width, float height);

/**
 * Retreives coordinate from a rectangle region object.
**/
__TRAX_EXPORT void trax_region_get_rectangle(const trax_region* region, float* x, float* y, float* width, float* height);

/**
 * Creates a polygon region object for a given amout of points. Note that the coordinates of the points
 * are arbitrary and have to be set after allocation.
**/
__TRAX_EXPORT trax_region* trax_region_create_polygon(int count);

/**
 * Sets coordinates of a given point in the polygon.
**/
__TRAX_EXPORT void trax_region_set_polygon_point(trax_region* region, int index, float x, float y);

/**
 * Retrieves the coordinates of a specific point in the polygon.
**/
__TRAX_EXPORT void trax_region_get_polygon_point(const trax_region* region, int index, float* x, float* y);

/**
 * Returns the number of points in the polygon.
**/
__TRAX_EXPORT int trax_region_get_polygon_count(const trax_region* region);

/**
 * Creates a mask region object for a given amout of points. Note that the mask data is not initialized.
**/
__TRAX_EXPORT trax_region* trax_region_create_mask(int x, int y, int width, int height);

/**
 * Returns the header data of a mask region.
**/
__TRAX_EXPORT void trax_region_get_mask_header(const trax_region* region, int* x, int* y, int* width, int* height);

/**
 * Returns a pointer for a writeable row in a data array of a mask.
**/
__TRAX_EXPORT char* trax_region_write_mask_row(trax_region* region, int row);

/**
 * Returns a read-only pointer for a row in a data array of a mask.
**/
__TRAX_EXPORT const char* trax_region_get_mask_row(const trax_region* region, int row);

/**
 * Calculates a bounding box region that bounds the input region.
 **/
__TRAX_EXPORT trax_bounds trax_region_bounds(const trax_region* region);

/**
 * Calculates if the region contains a given point.
 **/
__TRAX_EXPORT int trax_region_contains(const trax_region* region, float x, float y);

/**
 * Clones a region object.
 **/
__TRAX_EXPORT trax_region* trax_region_clone(const trax_region* region);

/**
 * Converts region between different formats (if possible).
 **/
__TRAX_EXPORT trax_region* trax_region_convert(const trax_region* region, int format);

/**
 * Calculates the spatial Jaccard index for two regions (overlap).
 **/
__TRAX_EXPORT float trax_region_overlap(const trax_region* a, const trax_region* b, const trax_bounds bounds);

/**
 * Encodes a region object to a string representation.
 **/
__TRAX_EXPORT char* trax_region_encode(const trax_region* region);

/**
 * Decodes string representation of a region to an object.
 **/
__TRAX_EXPORT trax_region* trax_region_decode(const char* data);

/**
* Allocate memory for storing multi object data.
**/
__TRAX_EXPORT trax_object_list* trax_object_list_create(int count);

/**
 * Destroy a object list object and clean up the memory, this also releases all objects.
 **/
__TRAX_EXPORT void trax_object_list_release(trax_object_list** list);

/**
 * Set a region of an object in a list. 
 **/
__TRAX_EXPORT void trax_object_list_set(trax_object_list* list, int index, trax_region* region);

/**
 * Set a region of an object in a list.
 **/
__TRAX_EXPORT trax_region* trax_object_list_get(trax_object_list* list, int index);

/**
 * Retrieve pointer to properties of a specific object in list.
 **/
__TRAX_EXPORT trax_properties* trax_object_list_properties(trax_object_list* list, int index);

/**
 * Get a number of objects in a list.
 **/
__TRAX_EXPORT int trax_object_list_count(const trax_object_list* list);

/**
 * Append one object list to another.
 **/
__TRAX_EXPORT int trax_object_list_append(trax_object_list* list, const trax_object_list* src);

/**
 * Destroy a properties object and clean up the memory.
 **/
__TRAX_EXPORT void trax_properties_release(trax_properties** properties);

/**
 * Clear a properties object.
 **/
__TRAX_EXPORT void trax_properties_clear(trax_properties* properties);

/**
 * Create a property object.
 **/
__TRAX_EXPORT trax_properties* trax_properties_create();

/**
 * Create a property object using values from extisting property object.
 **/
__TRAX_EXPORT trax_properties* trax_properties_copy(const trax_properties* original);

/**
 * Returns non-zero value if the entry exits and zero otherwise.
 **/
__TRAX_EXPORT int trax_properties_has(const trax_properties* properties, const char* key);

/**
 * Set a string property (the value string is cloned).
 **/
__TRAX_EXPORT void trax_properties_set(trax_properties* properties, const char* key, const char* value);

/**
 * Set an integer property. The value will be encoded as a string.
 **/
__TRAX_EXPORT void trax_properties_set_int(trax_properties* properties, const char* key, int value);

/**
 * Set a floating point value property. The value will be encoded as a string.
 **/
__TRAX_EXPORT void trax_properties_set_float(trax_properties* properties, const char* key, float value);

/**
 * Get a string property. The resulting string is a clone of the one stored so it should
 * be released when not needed anymore.
 **/
__TRAX_EXPORT char* trax_properties_get(const trax_properties* properties, const char* key);

/**
 * Get an integer property. A stored string value is converted to an integer. If this is not possible
 * or the property does not exist a given default value is returned.
 **/
__TRAX_EXPORT int trax_properties_get_int(const trax_properties* properties, const char* key, int def);

/**
 * Get a floating point value property. A stored string value is converted to an integer. If this is not possible
 * or the property does not exist a given default value is returned.
 **/
__TRAX_EXPORT float trax_properties_get_float(const trax_properties* properties, const char* key, float def);

/**
 * Get a number of all pairs in the properties object.
 **/
__TRAX_EXPORT int trax_properties_count(const trax_properties* properties);

/**
 * Iterate over the property set using a callback function. An optional pointer can be given and is forwarded
 * to the callback.
 **/
__TRAX_EXPORT void trax_properties_enumerate(const trax_properties* properties, trax_enumerator enumerator, const void* object);

/**
 * Append all properties from source to drain, optionally overwriting existing properties with same keys.
 **/
__TRAX_EXPORT void trax_properties_append(const trax_properties* source, trax_properties* drain, int overwrite);

/**
* Allocate memory for storing the input images
**/
__TRAX_EXPORT trax_image_list* trax_image_list_create();

/**
* Release image list structure, does not release any channel images
**/
__TRAX_EXPORT void trax_image_list_release(trax_image_list** list);

/**
* Cleans image list, releases all allocated channel images
**/
__TRAX_EXPORT void trax_image_list_clear(trax_image_list* list);

/**
* Get image at a specific channel
**/
__TRAX_EXPORT trax_image* trax_image_list_get(const trax_image_list* list, int channel);

/**
* Set image at a specific channel
**/
__TRAX_EXPORT void trax_image_list_set(trax_image_list* list, trax_image* image, int channel);

/**
* Size of the image list
**/
__TRAX_EXPORT int trax_image_list_count(int channels);

#ifdef __cplusplus
}

#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>

namespace trax {

class Image;
class ImageList;
class ObjectList;
class Region;
class Properties;

typedef trax_enumerator Enumerator;

class __TRAX_EXPORT Logging : public ::trax_logging {
public:
    Logging(trax_logging logging);
    Logging(trax_logger callback = NULL, void* data = NULL, int flags = 0);
    virtual ~Logging();
};

class __TRAX_EXPORT Bounds : public ::trax_bounds {
public:
    Bounds();
    Bounds(trax_bounds bounds);
    Bounds(float left, float top, float right, float bottom);
    virtual ~Bounds();
};

class __TRAX_EXPORT Wrapper {
public:
    virtual ~Wrapper();

    operator bool() const {
       return pn != NULL;
    }

protected:
    Wrapper();

    Wrapper(const Wrapper& count);

    Wrapper(Wrapper&& count);

    void swap(Wrapper& lhs);

    void acquire(const Wrapper& lhs);

    long claims() const;

    /**
     * Call after the wrapped pointer has been created or copied to increase
     * reference count;
    **/
    void acquire();

    /**
     * Call instead of releasing memory to decrease reference count. If the
     * reference count comes to zero then cleanup() is called.
    **/
    void release();

    virtual void cleanup() = 0;
    
private:

    long* pn;

};

class Handle;
class Client;
class ServerSOT;
class ServerMOT;

class __TRAX_EXPORT Metadata : public Wrapper {
friend class Handle;
friend class Client;
friend class ServerSOT;
friend class ServerMOT;
public:

    Metadata();

    Metadata(const Metadata& original);

    Metadata(Metadata&& original);

    Metadata(int region_formats, int image_formats, int channels = TRAX_CHANNEL_COLOR,
        std::string tracker_name = std::string(), std::string tracker_description = std::string(),
        std::string tracker_family = std::string(), int flags = 0);

    virtual ~Metadata();

    int image_formats() const;

    int region_formats() const;

    int channels() const;

    std::string tracker_name() const;

    std::string tracker_description() const;

    std::string tracker_family() const;

    std::string get_custom(const std::string key) const;

    void set_custom(const std::string key, const std::string value);

    Metadata& operator=(const Metadata& p) throw();

    Metadata& operator=(Metadata&& p) throw();

protected:

    Metadata(trax_metadata* metadata);

    virtual void cleanup();

    void wrap(trax_metadata* obj);

private:

    trax_metadata* metadata;

};

typedef Metadata Configuration;

class __TRAX_EXPORT Handle: public Wrapper {
public:
    /**
     * Closes communication, sends quit message if needed.
    **/
    virtual ~Handle();

    /**
     * Sets the parameter for the client or server instance.
    **/
    int set_parameter(int id, int value);

    /**
     * Gets the parameter for the client or server instance.
    **/
    int get_parameter(int id, int* value);

    const Metadata metadata();

    /**
     * Terminates session, sends quit message.
    **/
    bool terminate(const std::string reason = std::string());

    /**
     * Return last error string or empty string if no error has occured in last call to handle.
    **/
    std::string get_error();

    /**
     * Check if the handle is opened or not.
    **/
    bool is_alive();

protected:

    virtual void cleanup();

    void wrap(trax_handle* obj);

    Handle();

    Handle(const Handle& original);

    Handle(Handle&& original);

    trax_handle* handle;
};

class __TRAX_EXPORT Client: public Handle {
public:
    /**
     * Sets up the protocol for the client side and returns a handle object.
    **/
    Client(int input, int output, Logging logger);

    /**
     * Sets up the protocol for the client side and returns a handle object.
    **/
    Client(int server, Logging logger, int timeout = -1);

    virtual ~Client();

    /**
    * Waits for a valid protocol message from the server. Only works in single-object mode
    **/
    int wait(Region& region, Properties& properties);

    /**
    * Waits for a valid protocol message from the server. Returns the state of one or more objects.
    **/
    int wait(ObjectList& objects, Properties& properties);

    /**
    * Sends an initialize message.
    **/
    int initialize(const ImageList& image, const Region& region, const Properties& properties);

    /**
    * Sends an initialize message with one or more objects.
    **/
    int initialize(const ImageList& image, const ObjectList& objects, const Properties& properties);

    /**
    * Sends a frame message.
    **/
    int frame(const ImageList& image, const Properties& properties);

    /**
    * Sends a frame message that adds new objects (in multi-object mode).
    **/
    int frame(const ImageList& image, const ObjectList& objects, const Properties& properties);

protected:

    using Handle::cleanup;

private:

    Client& operator=(Client p) throw();

};

class __TRAX_EXPORT ServerMOT: public Handle {
public:

    /**
     * Sets up the protocol for the server side and returns a handle object.
    **/
    ServerMOT(Metadata metadata, Logging log);

    virtual ~ServerMOT();

    /**
     * Waits for a valid protocol message from the client.
    **/
    int wait(ImageList& image, Region& region, Properties& properties);

    /**
     * Sends a status reply to the client.
    **/
    int reply(const Region& region, const Properties& properties);

    /**
     * Waits for a valid protocol message from the client.
    **/
    int wait(ImageList& image, ObjectList& objects, Properties& properties);

    /**
     * Sends a status reply to the client.
    **/
    int reply(const ObjectList& objects);

private:
    ServerMOT& operator=(ServerMOT p) throw();

};

class __TRAX_EXPORT ServerSOT: public Handle {
public:

    /**
     * Sets up the protocol for the server side and returns a handle object.
    **/
    ServerSOT(Metadata metadata, Logging log);

    virtual ~ServerSOT();

    /**
     * Waits for a valid protocol message from the client.
    **/
    int wait(ImageList& image, Region& region, Properties& properties);

    /**
     * Sends a status reply to the client.
    **/
    int reply(const Region& region, const Properties& properties);

private:
    ServerSOT& operator=(ServerSOT p) throw();

};

class __TRAX_EXPORT Image : public Wrapper {
friend class Client;
friend class ServerSOT;
friend class ServerMOT;
friend class ImageList;
friend class ObjectList;
public:

    Image();

    Image(const Image& original);

    Image(Image&& original);

    /**
     * Creates a file-system path image description.
    **/
    static Image create_path(const std::string& path);

    /**
     * Creates a URL path image description.
    **/
    static Image create_url(const std::string& url);

    /**
     * Creates a raw buffer image description.
    **/
    static Image create_memory(int width, int height, int format);

    /**
     * Creates a file buffer image description.
    **/
    static Image create_buffer(int length, const char* data);

    /**
     * Releases image structure, frees allocated memory.
    **/
    virtual ~Image();

    /**
     * Returns a type of the image handle.
    **/
    int type() const;

    /**
     * Checks if image container is empty.
    **/
    bool empty() const;

    /**
     * Returns a file path from a file-system path image description. This function
     * returns a pointer to the internal data which should not be modified.
    **/
    const std::string get_path() const;

    /**
     * Returns a file path from a URL path image description. This function
     * returns a pointer to the internal data which should not be modified.
    **/
    const std::string get_url() const;

    /**
     * Returns the header data of a memory image.
    **/
    void get_memory_header(int* width, int* height, int* format) const;

    /**
     * Returns a pointer for a writeable row in a data array of an image.
    **/
    char* write_memory_row(int row);

    /**
     * Returns a read-only pointer for a row in a data array of an image.
    **/
    const char* get_memory_row(int row) const;

    /**
     * Returns a file buffer and its length. This function
     * returns a pointer to the internal data which should not be modified.
    **/
    const char* get_buffer(int* length, int* format) const;

    Image& operator=(const Image& lhs) throw();

    Image& operator=(Image&& lhs) throw();


protected:

    virtual void cleanup();

    void wrap(trax_image* obj);

private:

    trax_image* image;

};

class __TRAX_EXPORT ImageList : public Wrapper {
friend class Client;
friend class ServerSOT;
friend class ServerMOT;
public:

    ImageList();

    ImageList(const ImageList& original);

    ImageList(ImageList&& original);

    /**
     * Releases image list structure, frees allocated memory.
    **/
    virtual ~ImageList();

    /**
    * Get image at a specific channel
    **/
    Image get(int channel_num) const;

    /**
    * Test if list contains a specific channel
    **/
    bool has(int channel_num) const;

    /**
    * Set image at a specific channel
    **/
    void set(Image image, int channel_num);

    int size() const;

    ImageList& operator=(const ImageList& lhs) throw();

    ImageList& operator=(ImageList&& lhs) throw();

protected:

    virtual void cleanup();

    void wrap(trax_image_list* obj);

private:

    std::vector<Image> images;

    trax_image_list* list;

};

class __TRAX_EXPORT ObjectList : public Wrapper {
friend class Client;
friend class ServerSOT;
friend class ServerMOT;
public:
    ObjectList();

    ObjectList(int count);

    ObjectList(const ObjectList& original);

    ObjectList(ObjectList&& original);

    /**
     * Releases object list and frees allocated memory.
    **/
    virtual ~ObjectList();

    /**
    * Get object region at a specific index
    **/
    Region get(int index) const;

    /**
    * Set object region at a specific index
    **/
    void set(int index, Region region);

    /**
    * Get object properties at a specific index
    **/
    Properties properties(int index) const;

    int size() const;

    ObjectList& operator=(const ObjectList& lhs) throw();

    ObjectList& operator=(ObjectList&& lhs) throw();

protected:

    virtual void cleanup();

    void wrap(trax_object_list* obj);

private:

    std::vector<Region> _regions;
    std::vector<Properties> _properties;

    trax_object_list* list;

};

class __TRAX_EXPORT Region : public Wrapper {
friend class Client;
friend class ServerSOT;
friend class ServerMOT;
friend class ObjectList;
public:

    Region();

    Region(const Region& original);

    Region(Region&& original);


    /**
     * Creates a special region object. Only one paramter (region code) required.
    **/
    static Region create_special(int code);

    /**
     * Creates a rectangle region.
    **/
    static Region create_rectangle(float x, float y, float width, float height);

    /**
     * Creates a polygon region object for a given amout of points. Note that the coordinates of the points
     * are arbitrary and have to be set after allocation.
    **/
    static Region create_polygon(int count);

    /**
     * Creates a mask region object for a given amout of points. Note that the mask data is not initialized.
    **/
    static Region create_mask(int x, int y, int width, int height);

    /**
     * Releases region, frees allocated memory.
    **/
    virtual ~Region();

    /**
     * Returns type identifier of the region object.
    **/
    int type() const;

    /**
     * Checks if region container is empty.
    **/
    bool empty() const;

    /**
     * Sets the code of a special region.
    **/
    void set(int code);

    /**
     * Returns a code of a special region object.
    **/
    int get() const;

    /**
     * Sets the coordinates for a rectangle region.
    **/
    void set(float x, float y, float width, float height);

    /**
     * Retreives coordinate from a rectangle region object.
    **/
    void get(float* x, float* y, float* width, float* height) const;

    /**
     * Sets coordinates of a given point in the polygon.
    **/
    void set_polygon_point(int index, float x, float y);

    /**
     * Retrieves the coordinates of a specific point in the polygon.
    **/
    void get_polygon_point(int index, float* x, float* y) const;

    /**
     * Returns the number of points in the polygon.
    **/
    int get_polygon_count() const;

    /**
     * Returns the header data of a mask region.
    **/
    void get_mask_header(int* x, int* y, int* width, int* height) const;

    /**
     * Returns a pointer for a writeable row in a data array of a mask.
    **/
    char* write_mask_row(int row);

    /**
     * Returns a read-only pointer for a row in a data array of a mask.
    **/
    const char* get_mask_row(int row) const;

    /**
     * Computes bounds of a region.
     **/
    Bounds bounds() const;

    bool contains(float x, float y) const;

    Region convert(int type) const;

    float overlap(const Region& region, const Bounds& bounds = Bounds()) const;

    operator std::string () const;

    friend __TRAX_EXPORT std::ostream& operator<< (std::ostream& output, const Region& region);

    friend __TRAX_EXPORT std::istream& operator>> (std::istream& input, Region &D);

    Region& operator=(const Region& lhs) throw();

    Region& operator=(Region&& lhs) throw();

protected:

    virtual void cleanup();

    void wrap(trax_region* obj);

private:

    trax_region* region;
};

class __TRAX_EXPORT Properties : public Wrapper {
friend class Client;
friend class ServerSOT;
friend class ServerMOT;
friend class ObjectList;
public:

    /**
     * Create a property object.
     **/
    Properties();

    /**
     * A copy constructor.
     **/
    Properties(const Properties& original);

    /**
     * A move constructor.
     **/
    Properties(Properties&& original);

    /**
     * Destroy a properties object and clean up the memory.
     **/
    virtual ~Properties();

    /**
     * Return number of property pairs.
     **/
    int size() const;

    /**
     * Clear a properties object.
     **/
    void clear();

    /**
     * Set a string property (the value string is cloned).
     **/
    void set(const std::string key, const std::string value);

    /**
     * Set an integer property. The value will be encoded as a string.
     **/
    void set(const std::string key, int value);

    /**
     * Set a floating point value property. The value will be encoded as a string.
     **/
    void set(const std::string key, float value);

    /**
     * Get a string property.
     **/
    std::string get(const std::string key, const std::string& def = std::string()) const;

    std::string get(const std::string key, const char* def = NULL) const;

    /**
     * Get an integer property. A stored string value is converted to an integer. If this is not possible
     * or the property does not exist a given default value is returned.
     **/
    int get(const std::string key, int def) const;

    /**
     * Get a floating point value property. A stored string value is converted to a float. If this is not possible
     * or the property does not exist a given default value is returned.
     **/
    float get(const std::string key, float def) const;

    double get(const std::string key, double def) const;

    /**
     * Get a boolean point value property. A stored string value is converted to an integer and checked if it is zero. If this is not possible
     * or the property does not exist a given default value is returned.
     **/
    bool get(const std::string key, bool def) const;

    /**
     * Iterate over the property set using a callback function. An optional pointer can be given and is forwarded
     * to the callback.
     **/
    void enumerate(Enumerator enumerator, void* object);

    void from_map(const std::map<std::string, std::string>& m);

    void to_map(std::map<std::string, std::string>& m) const;

    void to_vector(std::vector<std::string>& v) const;

    friend __TRAX_EXPORT std::ostream& operator<< (std::ostream& output, const Properties& properties);

    Properties& operator=(const Properties& lhs) throw();

    Properties& operator=(Properties&& lhs) throw();

protected:

    virtual void cleanup();

    void wrap(trax_properties* obj);

private:

    void ensure_unique();

    trax_properties* properties;

};

#ifdef TRAX_LEGACY_SINGLE
typedef ServerSOT Server;
#else
typedef ServerMOT Server;
#endif

}

#endif

#endif
