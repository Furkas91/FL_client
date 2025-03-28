# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: fl_service_router.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='fl_service_router.proto',
  package='org.etu.fl',
  syntax='proto3',
  serialized_options=b'\n\017org.etu.fl.grpcP\001',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x17\x66l_service_router.proto\x12\norg.etu.fl\x1a\x1cgoogle/protobuf/struct.proto\"\xb0\x03\n\nDescriptor\x12,\n\x06isNull\x18\x01 \x01(\x0e\x32\x1a.google.protobuf.NullValueH\x00\x12\x14\n\nbool_value\x18\x02 \x01(\x08H\x00\x12\x14\n\nbyte_value\x18\x03 \x01(\x0cH\x00\x12\x15\n\x0bshort_value\x18\x04 \x01(\x0cH\x00\x12\x13\n\tint_value\x18\x05 \x01(\x05H\x00\x12\x14\n\nlong_value\x18\x06 \x01(\x03H\x00\x12\x15\n\x0b\x66loat_value\x18\x07 \x01(\x02H\x00\x12\x16\n\x0c\x64ouble_value\x18\x08 \x01(\x01H\x00\x12\x16\n\x0cstring_value\x18\t \x01(\tH\x00\x12\x31\n\x0b\x65numeration\x18\n \x01(\x0b\x32\x1a.org.etu.fl.EnumDescriptorH\x00\x12*\n\x04list\x18\x0b \x01(\x0b\x32\x1a.org.etu.fl.ListDescriptorH\x00\x12(\n\x03map\x18\x0c \x01(\x0b\x32\x19.org.etu.fl.MapDescriptorH\x00\x12.\n\x06object\x18\r \x01(\x0b\x32\x1c.org.etu.fl.ObjectDescriptorH\x00\x42\x06\n\x04kind\"V\n\x0e\x45numDescriptor\x12\x11\n\tenum_name\x18\x01 \x01(\t\x12\x18\n\x10\x65num_value_index\x18\x02 \x01(\x05\x12\x17\n\x0f\x65num_value_name\x18\x03 \x01(\t\"=\n\x0eListDescriptor\x12+\n\x0b\x64\x65scriptors\x18\x01 \x03(\x0b\x32\x16.org.etu.fl.Descriptor\"@\n\rMapDescriptor\x12/\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\x1e.org.etu.fl.MapDescriptorEntry\"`\n\x12MapDescriptorEntry\x12#\n\x03key\x18\x01 \x01(\x0b\x32\x16.org.etu.fl.Descriptor\x12%\n\x05value\x18\x02 \x01(\x0b\x32\x16.org.etu.fl.Descriptor\"\xa7\x01\n\x10ObjectDescriptor\x12\x12\n\nclass_name\x18\x01 \x01(\t\x12\x38\n\x06\x66ields\x18\x02 \x03(\x0b\x32(.org.etu.fl.ObjectDescriptor.FieldsEntry\x1a\x45\n\x0b\x46ieldsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12%\n\x05value\x18\x02 \x01(\x0b\x32\x16.org.etu.fl.Descriptor:\x02\x38\x01\"(\n\x12RequestedServiceID\x12\x12\n\nservice_id\x18\x01 \x01(\t\"\xba\x01\n\x08Schedule\x12$\n\x04node\x18\x01 \x01(\x0b\x32\x16.org.etu.fl.Descriptor\x12/\n\x0f\x61lgorithm_block\x18\x02 \x01(\x0b\x32\x16.org.etu.fl.Descriptor\x12+\n\x0b\x61ssignments\x18\x03 \x01(\x0b\x32\x16.org.etu.fl.Descriptor\x12*\n\x0csubschedules\x18\x04 \x03(\x0b\x32\x14.org.etu.fl.Schedule\"\x9e\x01\n\x12\x45xecutionContainer\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12(\n\x08settings\x18\x02 \x01(\x0b\x32\x16.org.etu.fl.Descriptor\x12%\n\x05model\x18\x03 \x01(\x0b\x32\x16.org.etu.fl.Descriptor\x12&\n\x08schedule\x18\x04 \x01(\x0b\x32\x14.org.etu.fl.Schedule\";\n\x10\x45rrorDescription\x12\x12\n\nerror_code\x18\x01 \x01(\r\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\"u\n\x0f\x45xecutionResult\x12\'\n\x05model\x18\x01 \x01(\x0b\x32\x16.org.etu.fl.DescriptorH\x00\x12\x31\n\terror_msg\x18\x02 \x01(\x0b\x32\x1c.org.etu.fl.ErrorDescriptionH\x00\x42\x06\n\x04kind2\xb7\x01\n\x0f\x46LRouterService\x12T\n\x1aReceiveFLServiceDescriptor\x12\x1e.org.etu.fl.RequestedServiceID\x1a\x16.org.etu.fl.Descriptor\x12N\n\x0f\x45xecuteSchedule\x12\x1e.org.etu.fl.ExecutionContainer\x1a\x1b.org.etu.fl.ExecutionResultB\x13\n\x0forg.etu.fl.grpcP\x01\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_struct__pb2.DESCRIPTOR,])




_DESCRIPTOR = _descriptor.Descriptor(
  name='Descriptor',
  full_name='org.etu.fl.Descriptor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='isNull', full_name='org.etu.fl.Descriptor.isNull', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bool_value', full_name='org.etu.fl.Descriptor.bool_value', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='byte_value', full_name='org.etu.fl.Descriptor.byte_value', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='short_value', full_name='org.etu.fl.Descriptor.short_value', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='int_value', full_name='org.etu.fl.Descriptor.int_value', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='long_value', full_name='org.etu.fl.Descriptor.long_value', index=5,
      number=6, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='float_value', full_name='org.etu.fl.Descriptor.float_value', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='double_value', full_name='org.etu.fl.Descriptor.double_value', index=7,
      number=8, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='string_value', full_name='org.etu.fl.Descriptor.string_value', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='enumeration', full_name='org.etu.fl.Descriptor.enumeration', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='list', full_name='org.etu.fl.Descriptor.list', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='map', full_name='org.etu.fl.Descriptor.map', index=11,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='object', full_name='org.etu.fl.Descriptor.object', index=12,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='kind', full_name='org.etu.fl.Descriptor.kind',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=70,
  serialized_end=502,
)


_ENUMDESCRIPTOR = _descriptor.Descriptor(
  name='EnumDescriptor',
  full_name='org.etu.fl.EnumDescriptor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='enum_name', full_name='org.etu.fl.EnumDescriptor.enum_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='enum_value_index', full_name='org.etu.fl.EnumDescriptor.enum_value_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='enum_value_name', full_name='org.etu.fl.EnumDescriptor.enum_value_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=504,
  serialized_end=590,
)


_LISTDESCRIPTOR = _descriptor.Descriptor(
  name='ListDescriptor',
  full_name='org.etu.fl.ListDescriptor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='descriptors', full_name='org.etu.fl.ListDescriptor.descriptors', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=592,
  serialized_end=653,
)


_MAPDESCRIPTOR = _descriptor.Descriptor(
  name='MapDescriptor',
  full_name='org.etu.fl.MapDescriptor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='entries', full_name='org.etu.fl.MapDescriptor.entries', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=655,
  serialized_end=719,
)


_MAPDESCRIPTORENTRY = _descriptor.Descriptor(
  name='MapDescriptorEntry',
  full_name='org.etu.fl.MapDescriptorEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='org.etu.fl.MapDescriptorEntry.key', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='org.etu.fl.MapDescriptorEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=721,
  serialized_end=817,
)


_OBJECTDESCRIPTOR_FIELDSENTRY = _descriptor.Descriptor(
  name='FieldsEntry',
  full_name='org.etu.fl.ObjectDescriptor.FieldsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='org.etu.fl.ObjectDescriptor.FieldsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='org.etu.fl.ObjectDescriptor.FieldsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=918,
  serialized_end=987,
)

_OBJECTDESCRIPTOR = _descriptor.Descriptor(
  name='ObjectDescriptor',
  full_name='org.etu.fl.ObjectDescriptor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='class_name', full_name='org.etu.fl.ObjectDescriptor.class_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='fields', full_name='org.etu.fl.ObjectDescriptor.fields', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_OBJECTDESCRIPTOR_FIELDSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=820,
  serialized_end=987,
)


_REQUESTEDSERVICEID = _descriptor.Descriptor(
  name='RequestedServiceID',
  full_name='org.etu.fl.RequestedServiceID',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='service_id', full_name='org.etu.fl.RequestedServiceID.service_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=989,
  serialized_end=1029,
)


_SCHEDULE = _descriptor.Descriptor(
  name='Schedule',
  full_name='org.etu.fl.Schedule',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='node', full_name='org.etu.fl.Schedule.node', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='algorithm_block', full_name='org.etu.fl.Schedule.algorithm_block', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='assignments', full_name='org.etu.fl.Schedule.assignments', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='subschedules', full_name='org.etu.fl.Schedule.subschedules', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1032,
  serialized_end=1218,
)


_EXECUTIONCONTAINER = _descriptor.Descriptor(
  name='ExecutionContainer',
  full_name='org.etu.fl.ExecutionContainer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='task_id', full_name='org.etu.fl.ExecutionContainer.task_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='settings', full_name='org.etu.fl.ExecutionContainer.settings', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model', full_name='org.etu.fl.ExecutionContainer.model', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='schedule', full_name='org.etu.fl.ExecutionContainer.schedule', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1221,
  serialized_end=1379,
)


_ERRORDESCRIPTION = _descriptor.Descriptor(
  name='ErrorDescription',
  full_name='org.etu.fl.ErrorDescription',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='error_code', full_name='org.etu.fl.ErrorDescription.error_code', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='description', full_name='org.etu.fl.ErrorDescription.description', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1381,
  serialized_end=1440,
)


_EXECUTIONRESULT = _descriptor.Descriptor(
  name='ExecutionResult',
  full_name='org.etu.fl.ExecutionResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='org.etu.fl.ExecutionResult.model', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='error_msg', full_name='org.etu.fl.ExecutionResult.error_msg', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='kind', full_name='org.etu.fl.ExecutionResult.kind',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=1442,
  serialized_end=1559,
)

_DESCRIPTOR.fields_by_name['isNull'].enum_type = google_dot_protobuf_dot_struct__pb2._NULLVALUE
_DESCRIPTOR.fields_by_name['enumeration'].message_type = _ENUMDESCRIPTOR
_DESCRIPTOR.fields_by_name['list'].message_type = _LISTDESCRIPTOR
_DESCRIPTOR.fields_by_name['map'].message_type = _MAPDESCRIPTOR
_DESCRIPTOR.fields_by_name['object'].message_type = _OBJECTDESCRIPTOR
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['isNull'])
_DESCRIPTOR.fields_by_name['isNull'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['bool_value'])
_DESCRIPTOR.fields_by_name['bool_value'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['byte_value'])
_DESCRIPTOR.fields_by_name['byte_value'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['short_value'])
_DESCRIPTOR.fields_by_name['short_value'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['int_value'])
_DESCRIPTOR.fields_by_name['int_value'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['long_value'])
_DESCRIPTOR.fields_by_name['long_value'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['float_value'])
_DESCRIPTOR.fields_by_name['float_value'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['double_value'])
_DESCRIPTOR.fields_by_name['double_value'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['string_value'])
_DESCRIPTOR.fields_by_name['string_value'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['enumeration'])
_DESCRIPTOR.fields_by_name['enumeration'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['list'])
_DESCRIPTOR.fields_by_name['list'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['map'])
_DESCRIPTOR.fields_by_name['map'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_DESCRIPTOR.oneofs_by_name['kind'].fields.append(
  _DESCRIPTOR.fields_by_name['object'])
_DESCRIPTOR.fields_by_name['object'].containing_oneof = _DESCRIPTOR.oneofs_by_name['kind']
_LISTDESCRIPTOR.fields_by_name['descriptors'].message_type = _DESCRIPTOR
_MAPDESCRIPTOR.fields_by_name['entries'].message_type = _MAPDESCRIPTORENTRY
_MAPDESCRIPTORENTRY.fields_by_name['key'].message_type = _DESCRIPTOR
_MAPDESCRIPTORENTRY.fields_by_name['value'].message_type = _DESCRIPTOR
_OBJECTDESCRIPTOR_FIELDSENTRY.fields_by_name['value'].message_type = _DESCRIPTOR
_OBJECTDESCRIPTOR_FIELDSENTRY.containing_type = _OBJECTDESCRIPTOR
_OBJECTDESCRIPTOR.fields_by_name['fields'].message_type = _OBJECTDESCRIPTOR_FIELDSENTRY
_SCHEDULE.fields_by_name['node'].message_type = _DESCRIPTOR
_SCHEDULE.fields_by_name['algorithm_block'].message_type = _DESCRIPTOR
_SCHEDULE.fields_by_name['assignments'].message_type = _DESCRIPTOR
_SCHEDULE.fields_by_name['subschedules'].message_type = _SCHEDULE
_EXECUTIONCONTAINER.fields_by_name['settings'].message_type = _DESCRIPTOR
_EXECUTIONCONTAINER.fields_by_name['model'].message_type = _DESCRIPTOR
_EXECUTIONCONTAINER.fields_by_name['schedule'].message_type = _SCHEDULE
_EXECUTIONRESULT.fields_by_name['model'].message_type = _DESCRIPTOR
_EXECUTIONRESULT.fields_by_name['error_msg'].message_type = _ERRORDESCRIPTION
_EXECUTIONRESULT.oneofs_by_name['kind'].fields.append(
  _EXECUTIONRESULT.fields_by_name['model'])
_EXECUTIONRESULT.fields_by_name['model'].containing_oneof = _EXECUTIONRESULT.oneofs_by_name['kind']
_EXECUTIONRESULT.oneofs_by_name['kind'].fields.append(
  _EXECUTIONRESULT.fields_by_name['error_msg'])
_EXECUTIONRESULT.fields_by_name['error_msg'].containing_oneof = _EXECUTIONRESULT.oneofs_by_name['kind']
DESCRIPTOR.message_types_by_name['Descriptor'] = _DESCRIPTOR
DESCRIPTOR.message_types_by_name['EnumDescriptor'] = _ENUMDESCRIPTOR
DESCRIPTOR.message_types_by_name['ListDescriptor'] = _LISTDESCRIPTOR
DESCRIPTOR.message_types_by_name['MapDescriptor'] = _MAPDESCRIPTOR
DESCRIPTOR.message_types_by_name['MapDescriptorEntry'] = _MAPDESCRIPTORENTRY
DESCRIPTOR.message_types_by_name['ObjectDescriptor'] = _OBJECTDESCRIPTOR
DESCRIPTOR.message_types_by_name['RequestedServiceID'] = _REQUESTEDSERVICEID
DESCRIPTOR.message_types_by_name['Schedule'] = _SCHEDULE
DESCRIPTOR.message_types_by_name['ExecutionContainer'] = _EXECUTIONCONTAINER
DESCRIPTOR.message_types_by_name['ErrorDescription'] = _ERRORDESCRIPTION
DESCRIPTOR.message_types_by_name['ExecutionResult'] = _EXECUTIONRESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Descriptor = _reflection.GeneratedProtocolMessageType('Descriptor', (_message.Message,), {
  'DESCRIPTOR' : _DESCRIPTOR,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.Descriptor)
  })
_sym_db.RegisterMessage(Descriptor)

EnumDescriptor = _reflection.GeneratedProtocolMessageType('EnumDescriptor', (_message.Message,), {
  'DESCRIPTOR' : _ENUMDESCRIPTOR,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.EnumDescriptor)
  })
_sym_db.RegisterMessage(EnumDescriptor)

ListDescriptor = _reflection.GeneratedProtocolMessageType('ListDescriptor', (_message.Message,), {
  'DESCRIPTOR' : _LISTDESCRIPTOR,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.ListDescriptor)
  })
_sym_db.RegisterMessage(ListDescriptor)

MapDescriptor = _reflection.GeneratedProtocolMessageType('MapDescriptor', (_message.Message,), {
  'DESCRIPTOR' : _MAPDESCRIPTOR,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.MapDescriptor)
  })
_sym_db.RegisterMessage(MapDescriptor)

MapDescriptorEntry = _reflection.GeneratedProtocolMessageType('MapDescriptorEntry', (_message.Message,), {
  'DESCRIPTOR' : _MAPDESCRIPTORENTRY,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.MapDescriptorEntry)
  })
_sym_db.RegisterMessage(MapDescriptorEntry)

ObjectDescriptor = _reflection.GeneratedProtocolMessageType('ObjectDescriptor', (_message.Message,), {

  'FieldsEntry' : _reflection.GeneratedProtocolMessageType('FieldsEntry', (_message.Message,), {
    'DESCRIPTOR' : _OBJECTDESCRIPTOR_FIELDSENTRY,
    '__module__' : 'fl_service_router_pb2'
    # @@protoc_insertion_point(class_scope:org.etu.fl.ObjectDescriptor.FieldsEntry)
    })
  ,
  'DESCRIPTOR' : _OBJECTDESCRIPTOR,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.ObjectDescriptor)
  })
_sym_db.RegisterMessage(ObjectDescriptor)
_sym_db.RegisterMessage(ObjectDescriptor.FieldsEntry)

RequestedServiceID = _reflection.GeneratedProtocolMessageType('RequestedServiceID', (_message.Message,), {
  'DESCRIPTOR' : _REQUESTEDSERVICEID,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.RequestedServiceID)
  })
_sym_db.RegisterMessage(RequestedServiceID)

Schedule = _reflection.GeneratedProtocolMessageType('Schedule', (_message.Message,), {
  'DESCRIPTOR' : _SCHEDULE,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.Schedule)
  })
_sym_db.RegisterMessage(Schedule)

ExecutionContainer = _reflection.GeneratedProtocolMessageType('ExecutionContainer', (_message.Message,), {
  'DESCRIPTOR' : _EXECUTIONCONTAINER,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.ExecutionContainer)
  })
_sym_db.RegisterMessage(ExecutionContainer)

ErrorDescription = _reflection.GeneratedProtocolMessageType('ErrorDescription', (_message.Message,), {
  'DESCRIPTOR' : _ERRORDESCRIPTION,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.ErrorDescription)
  })
_sym_db.RegisterMessage(ErrorDescription)

ExecutionResult = _reflection.GeneratedProtocolMessageType('ExecutionResult', (_message.Message,), {
  'DESCRIPTOR' : _EXECUTIONRESULT,
  '__module__' : 'fl_service_router_pb2'
  # @@protoc_insertion_point(class_scope:org.etu.fl.ExecutionResult)
  })
_sym_db.RegisterMessage(ExecutionResult)


DESCRIPTOR._options = None
_OBJECTDESCRIPTOR_FIELDSENTRY._options = None

_FLROUTERSERVICE = _descriptor.ServiceDescriptor(
  name='FLRouterService',
  full_name='org.etu.fl.FLRouterService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1562,
  serialized_end=1745,
  methods=[
  _descriptor.MethodDescriptor(
    name='ReceiveFLServiceDescriptor',
    full_name='org.etu.fl.FLRouterService.ReceiveFLServiceDescriptor',
    index=0,
    containing_service=None,
    input_type=_REQUESTEDSERVICEID,
    output_type=_DESCRIPTOR,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ExecuteSchedule',
    full_name='org.etu.fl.FLRouterService.ExecuteSchedule',
    index=1,
    containing_service=None,
    input_type=_EXECUTIONCONTAINER,
    output_type=_EXECUTIONRESULT,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_FLROUTERSERVICE)

DESCRIPTOR.services_by_name['FLRouterService'] = _FLROUTERSERVICE

# @@protoc_insertion_point(module_scope)
