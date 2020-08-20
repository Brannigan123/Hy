package hy.util;

import java.util.LinkedList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import lombok.experimental.UtilityClass;

@UtilityClass
public class StreamUtil {

	public <E> LinkedList<E> toList(Stream<E> stream) {
		return stream.collect(Collectors.toCollection(LinkedList::new));
	}

	public <T, U> Stream<U> filter(Stream<T> incoming, Class<U> clazz) {
		return incoming.filter(clazz::isInstance).map(clazz::cast);
	}
}
